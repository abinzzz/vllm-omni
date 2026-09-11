# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify MammothModa2 image-cache reuse in a persistent AR-only engine.

Run from the repository root with ``python -m benchmarks.mammoth_moda2_cache``.
Worker instrumentation is enabled only by this benchmark's deployment config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import psutil
import yaml
from PIL import Image, ImageChops


class CacheProbeWorkerExtension:
    """Observe real encoding without altering the scheduler or cache policy."""

    def mammoth_cache_probe(self, action="snapshot"):
        import torch

        runner = self.model_runner
        if action == "start":
            if hasattr(self, "_mammoth_probe_counts"):
                raise RuntimeError("Cache probe is already installed")
            counts = self._mammoth_probe_counts = {"encoder_calls": 0, "encoded_images": 0}
            self._mammoth_probe_new_keys = []
            original = runner.model.embed_multimodal
            original_store = runner._cache_encoder_output

            def encode(**kwargs):
                outputs = original(**kwargs)
                counts["encoder_calls"] += 1
                counts["encoded_images"] += len(outputs)
                return outputs

            runner.model.embed_multimodal = encode

            def store(key, *args, **kwargs):
                original_store(key, *args, **kwargs)
                self._mammoth_probe_new_keys.append(key)

            runner._cache_encoder_output = store
            torch.accelerator.reset_peak_memory_stats()
        elif action != "snapshot":
            raise ValueError(action)

        torch.accelerator.synchronize()
        features = {}
        storages = {}
        for key, value in runner.encoder_cache.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Unsupported encoder-cache value: {type(value)}")
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
            raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            features[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        newly_encoded = [features[key] for key in self._mammoth_probe_new_keys if key in features]
        self._mammoth_probe_new_keys.clear()
        return {
            **self._mammoth_probe_counts,
            "encoder_cache_storage_bytes": sum(storages.values()),
            "encoder_features": features,
            "newly_encoded_features": newly_encoded,
            "gpu_allocated_bytes": torch.cuda.memory_allocated(),
            "gpu_reserved_bytes": torch.cuda.memory_reserved(),
            "gpu_peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
        }


def _worker_stats(omni, action="snapshot"):
    result = omni.engine.collective_rpc("mammoth_cache_probe", args=(action,), stage_ids=[0], timeout=60)
    while isinstance(result, list) and len(result) == 1:
        result = result[0]
    if not isinstance(result, dict) or "encoder_calls" not in result:
        raise RuntimeError(f"Expected one AR worker's cache statistics, got {result!r}")
    return result


def _processor_stats(omni):
    cache = omni.engine.input_processor.renderer.mm_processor_cache
    if cache is None:
        return {"hits": 0, "total": 0, "accounted_bytes": 0, "cache_type": None}
    stats = cache.make_stats()
    return {
        **stats._asdict(),
        # Sender-cache accounting represents retained receiver items, not P0 RSS.
        "accounted_bytes": cache._cache.currsize,
        "cache_type": type(cache).__name__,
    }


def _request(omni, model, image, prompt, max_tokens, processor_kwargs=None):
    from vllm import SamplingParams

    from vllm_omni.model_extras import build_x_to_text_prompt

    inputs, stop_ids = build_x_to_text_prompt(model_family="mammoth_moda2", model=model, prompt=prompt, has_image=True)
    inputs["multi_modal_data"] = {"image": image.copy()}
    if processor_kwargs:
        inputs["mm_processor_kwargs"] = processor_kwargs
    params = SamplingParams(temperature=0, seed=42, max_tokens=max_tokens, stop_token_ids=stop_ids)
    start = time.perf_counter()
    outputs = list(omni.generate([inputs], sampling_params_list=[params]))
    elapsed = time.perf_counter() - start
    completions = []
    for output in outputs:
        request_output = getattr(output, "request_output", output)
        completions.extend(getattr(request_output, "outputs", None) or [])
    if len(completions) != 1:
        raise RuntimeError(f"Expected one final completion, got {len(completions)}")
    completion = completions[0]
    return {
        "latency_s": elapsed,
        "token_ids": list(completion.token_ids),
        "text": completion.text,
        "finish_reason": completion.finish_reason,
    }


def _variant(image, index):
    result = ImageChops.offset(image, index + 1, index + 1)
    result.putpixel((0, 0), (index % 256, (index // 256) % 256, 137))
    return result


def _run_arm(args, image, enabled):
    from vllm_omni import Omni

    config = yaml.safe_load(Path(args.deploy_config).read_text())
    if config.get("pipeline") != "mammoth_moda2_ar" or len(config["stages"]) != 1:
        raise ValueError("Use a single-stage mammoth_moda2_ar deployment")
    stage = config["stages"][0]
    if stage.get("tensor_parallel_size", 1) != 1 or stage.get("num_replicas", 1) != 1:
        raise ValueError("This benchmark supports one AR worker only")
    stage.update(
        max_num_seqs=1,
        enable_prefix_caching=False,
        enforce_eager=True,
        mm_processor_cache_gb=args.cache_gb if enabled else 0,
        worker_extension_cls="benchmarks.mammoth_moda2_cache.CacheProbeWorkerExtension",
    )
    records = []
    with tempfile.TemporaryDirectory(prefix="mammoth-cache-") as directory:
        deploy = Path(directory, "deploy.yaml")
        deploy.write_text(yaml.safe_dump(config))
        start = time.perf_counter()
        omni = Omni(model=args.model, deploy_config=str(deploy), trust_remote_code=True, log_stats=False)
        startup_s = time.perf_counter() - start
        try:
            for index in range(args.warmup):
                _request(omni, args.model, _variant(image, 1000 + index), args.prompt, args.max_tokens)
            previous = _worker_stats(omni, "start")
            previous_processor = _processor_stats(omni)
            cases = [("first_a", image, args.prompt, None)]
            cases.extend((f"repeat_a_{i}", image, args.prompt, None) for i in range(args.repeats))
            cases.append(("changed_text", image, "Which text is visible in the image?", None))
            cases.append(
                ("changed_options", image, args.prompt, {"size": {"shortest_edge": 56**2, "longest_edge": 112**2}})
            )
            cases.extend((f"unique_{i}", _variant(image, i), args.prompt, None) for i in range(args.repeats))
            if args.eviction_requests:
                cases.extend(
                    (f"evict_{i}", _variant(image, 100 + i), args.prompt, None) for i in range(args.eviction_requests)
                )
                cases.append(("after_encoder_eviction", image, args.prompt, None))
            for name, case_image, prompt, kwargs in cases:
                output = _request(omni, args.model, case_image, prompt, args.max_tokens, kwargs)
                worker = _worker_stats(omni)
                processor = _processor_stats(omni)
                record = {
                    "case": name,
                    "image_sha256": hashlib.sha256(case_image.tobytes()).hexdigest(),
                    "processor_kwargs": kwargs or {},
                    **output,
                    "encoder_calls": worker["encoder_calls"] - previous["encoder_calls"],
                    "encoded_images": worker["encoded_images"] - previous["encoded_images"],
                    "processor_hits": processor["hits"] - previous_processor["hits"],
                    "processor_lookups": processor["total"] - previous_processor["total"],
                    "processor_cache": processor,
                    "worker": worker,
                    "frontend_rss_bytes": psutil.Process().memory_info().rss,
                }
                records.append(record)
                previous, previous_processor = worker, processor
                print(
                    f"cache={'on' if enabled else 'off'} {name}: {output['latency_s']:.3f}s, "
                    f"processor_hits={record['processor_hits']}, encoded_images={record['encoded_images']}",
                    flush=True,
                )
        finally:
            omni.close()
    return {"enabled": enabled, "startup_s": startup_s, "config": config, "records": records}


def _validate(arms):
    disabled, enabled = arms
    checks = []
    for base, cached in zip(disabled["records"], enabled["records"], strict=True):
        name = base["case"]
        reuse = name.startswith("repeat_a_") or name == "changed_text"
        processor_reuse = reuse or name == "after_encoder_eviction"
        checks.append(
            {"case": name, "check": "identical_generated_tokens", "passed": base["token_ids"] == cached["token_ids"]}
        )
        checks.append({"case": name, "check": "disabled_reencodes", "passed": base["encoded_images"] == 1})
        checks.append(
            {"case": name, "check": "expected_encoder_reuse", "passed": cached["encoded_images"] == (0 if reuse else 1)}
        )
        checks.append(
            {
                "case": name,
                "check": "expected_processor_reuse",
                "passed": (cached["processor_hits"] > 0) == processor_reuse,
            }
        )
        # The disabled arm uses request-scoped keys. Compare feature contents,
        # not identifiers, with the retained entries in the enabled arm.
        base_features = base["worker"]["newly_encoded_features"]
        cached_features = list(cached["worker"]["encoder_features"].values())
        checks.append(
            {
                "case": name,
                "check": "encoder_feature_equivalence",
                "passed": bool(base_features) and all(v in cached_features for v in base_features),
            }
        )
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="Describe the image in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cache-gb", type=float, default=1)
    parser.add_argument("--max-image-size", type=int, default=448)
    parser.add_argument(
        "--eviction-requests",
        type=int,
        default=0,
        help="Fill the encoder cache with distinct images, then require A to be re-encoded (0 disables this check).",
    )
    args = parser.parse_args()
    if min(args.repeats, args.warmup, args.max_tokens, args.max_image_size) < 1 or args.cache_gb <= 0:
        parser.error("counts, image size, and cache capacity must be positive")
    if args.eviction_requests < 0:
        parser.error("--eviction-requests must be nonnegative")
    image = Image.open(args.image).convert("RGB")
    image.thumbnail((args.max_image_size, args.max_image_size))
    model = Path(args.model)
    metadata = model / ".cache/huggingface/download/config.json.metadata"
    report = {
        "arguments": vars(args),
        "image_size": list(image.size),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_revision": metadata.read_text().splitlines()[0] if metadata.exists() else None,
        "model_config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
        "versions": {name: version(name) for name in ("vllm", "vllm-omni", "torch", "transformers")},
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True
        ).strip(),
        "timing_scope": "Omni.generate wall time; excludes startup, warmup, image I/O, prompt building and probe RPCs",
        "arms": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for enabled in (False, True):
        report["arms"].append(_run_arm(args, image, enabled))
        output.write_text(json.dumps(report, indent=2) + "\n")
    report["checks"] = _validate(report["arms"])
    report["passed"] = all(check["passed"] for check in report["checks"])
    report["latency_medians_s"] = {
        "on" if arm["enabled"] else "off": {
            group: statistics.median(r["latency_s"] for r in arm["records"] if r["case"].startswith(group))
            for group in ("repeat_a_", "unique_")
        }
        for arm in report["arms"]
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "latency_medians_s": report["latency_medians_s"]}, indent=2))
    if not report["passed"]:
        raise SystemExit("Cache validation failed; see the checks in the JSON report")


if __name__ == "__main__":
    main()
