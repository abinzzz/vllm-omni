# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-L40S Preview T2I regression with multimodal caching off/on.

Set MAMMOTH_MODA2_T2I_MODEL to a pinned local Preview checkpoint. Each cache
configuration uses a fresh engine; both AR and DiT run on one visible GPU.
"""

import hashlib
import json
import os
import subprocess
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "L40S"})
@pytest.mark.parametrize("resolution,num_steps", [(256, 5), (512, 50)], ids=["256px-5steps", "512px-50steps"])
def test_preview_t2i_preserves_output_with_mm_cache(tmp_path, resolution, num_steps):
    model = os.environ.get("MAMMOTH_MODA2_T2I_MODEL", "")
    if not model:
        pytest.skip("Set MAMMOTH_MODA2_T2I_MODEL to a pinned local Preview checkpoint")
    from vllm_omni import Omni
    from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
    from vllm_omni.model_extras.mammothmodal2_preview import build_text_to_image_prompt

    model_path = Path(model)
    config_path = model_path / "config.json"
    metadata = model_path / ".cache/huggingface/download/config.json.metadata"
    model_config = json.loads(config_path.read_text())
    assert model_config["llm_config"]["model_type"] == "mammothmoda2_qwen2_5_vl", "Use the Preview checkpoint"
    assert torch.cuda.is_available(), "This regression requires one CUDA GPU"
    repo_root = Path(__file__).resolve().parents[3]
    deploy_config = yaml.safe_load((repo_root / "benchmarks/mammoth_moda2_cache.yaml").read_text())
    deploy_config["pipeline"] = "mammoth_moda2"
    deploy_config["stages"].append(
        {
            "stage_id": 1,
            "devices": "0",
            "max_num_seqs": 1,
            "gpu_memory_utilization": 0.16,
            "enforce_eager": True,
            "trust_remote_code": True,
            "enable_prefix_caching": False,
        }
    )
    prompt = build_text_to_image_prompt(
        "A red ceramic mug on a white table, studio photograph.", None, height=resolution, width=resolution
    )
    report = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=repo_root, text=True),
        "model": str(model_path.resolve()),
        "model_revision": metadata.read_text().splitlines()[0] if metadata.exists() else None,
        "model_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": {name: version(name) for name in ("torch", "vllm", "vllm-omni", "transformers")},
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader"], text=True
        ).strip(),
        "prompt": prompt,
        "seed": 42,
        "resolution": resolution,
        "num_inference_steps": num_steps,
        "text_guidance_scale": 4.0,
        "arms": [],
    }
    images = []
    for enabled in (False, True):
        config = deepcopy(deploy_config)
        for stage in config["stages"]:
            stage["mm_processor_cache_gb"] = 1 if enabled else 0
        deploy = tmp_path / f"deploy-{'on' if enabled else 'off'}.yaml"
        deploy.write_text(yaml.safe_dump(config))
        omni = Omni(model=str(model_path), deploy_config=str(deploy), trust_remote_code=True, log_stats=False)
        try:
            params = deepcopy(omni.default_sampling_params_list)
            for item in params:
                item.seed = 42
                item.extra_args = {
                    "text_guidance_scale": 4.0,
                    "cfg_range": [0.0, 1.0],
                    "num_inference_steps": num_steps,
                }
            grid_size = resolution // 16
            params[0].max_tokens = grid_size * (grid_size + 1) + 1
            params[0].temperature = 0
            outputs = list(omni.generate(deepcopy(prompt), sampling_params_list=params))
            results = extract_images_from_outputs(outputs)
            assert len(results) == 1, "Expected one generated image"
            image = results[0].convert("RGB")
            assert image.size == (resolution, resolution)
            pixels = np.asarray(image)
            assert pixels.std() > 0, "Generated image is blank"
            path = tmp_path / f"preview-t2i-cache-{'on' if enabled else 'off'}.png"
            image.save(path)
            images.append(pixels)
            report["arms"].append(
                {
                    "enabled": enabled,
                    "config": config,
                    "image": path.name,
                    "sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                }
            )
            (tmp_path / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        finally:
            omni.close()
    report["identical_image_pixels"] = bool(np.array_equal(*images))
    (tmp_path / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.testing.assert_array_equal(*images, err_msg="Multimodal cache settings changed Preview text-to-image output")
