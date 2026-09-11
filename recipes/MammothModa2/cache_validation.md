# MammothModa2 multimodal cache validation

This recipe covers the AR-stage preprocessing and vision-encoder caches tracked
by [#7211](https://github.com/vllm-project/vllm-omni/issues/7211), under
[#7075](https://github.com/vllm-project/vllm-omni/issues/7075). It does not enable
AR prefix/KV caching, DiT caching, or cross-replica encoder sharing.

## Existing cache integration

MammothModa2 delegates image preprocessing and visual encoding to the inherited
Qwen implementations. The vLLM multimodal processor caches processed image
items and prompt updates. Its hashes include processing options, including
when a caller provides an image UUID. Image-only results can be reused when
the text changes, while tokenization and placeholder placement still follow
the current request.

The scheduler's encoder cache manager identifies visual features by their
multimodal identifiers. Finishing a request releases its references; an
unreferenced feature can remain available for another request until capacity
pressure evicts it. The runner owns the actual feature tensors. No additional
model-local image cache is required for the validated configurations.

`enable_prefix_caching: false` does not by itself disable these caches. In the
inspected vLLM 0.28.0 implementation, setting `mm_processor_cache_gb: 0` as well
causes the renderer to use request-local multimodal identifiers. This disables
cross-request encoder reuse too. The disabled benchmark arm is therefore a
baseline for both caches together, not an isolated preprocessing ablation.

User-provided UUIDs are trusted identities: do not reuse one UUID for different
image contents. For stage replicas, retain Omni's replica scoping so frontend
sender-cache hits do not omit tensors for an engine that never received them.

## Processor correctness tests

The tests use actual HF/vLLM processors and synthetic RGB images. They do not
load model weights. Download the configuration, preprocessor and tokenizer
assets, or use an existing checkpoint directory, then run:

```bash
MAMMOTH_MODA2_TEST_MODEL=/path/to/MammothModa2-Preview \
  python -m pytest tests/model_executor/models/mammoth_moda2/test_multimodal_cache.py -q

MAMMOTH_MODA2_TEST_MODEL=/path/to/MammothModa2-Dev \
  python -m pytest tests/model_executor/models/mammoth_moda2/test_multimodal_cache.py -q
```

The fixture deliberately skips when `MAMMOTH_MODA2_TEST_MODEL` is unset, so
ordinary unit-test collection does not download checkpoints. Install the
repository's test dependencies; for a focused run without pytest-xdist, add
`-o addopts=''`.

Coverage includes:

- Cache-disabled, miss and hit equality for token IDs, image tensors, grids,
  multimodal hashes and placeholder metadata.
- Actual image-preprocessor call counts for repeated images and changed text,
  using both string/token prompts and content hashes/explicit UUIDs.
- Changed image processing options that change the output grid and cache key.
- Distinct images, partial hits, reordered image lists and duplicate images.
- Cache clearing, bounded-capacity eviction and correct recomputation.
- Text-only requests preserve their output without image-cache lookups.

## Persistent-engine GPU validation

Use the editable checkout so the benchmark-only worker extension is importable
in spawned workers. The supplied configuration targets one L40S 48 GB and one
AR worker. Run from the repository root:

```bash
python -m benchmarks.mammoth_moda2_cache \
  --model /path/to/MammothModa2-Preview \
  --deploy-config benchmarks/mammoth_moda2_cache.yaml \
  --image /path/to/image.png \
  --repeats 7 --warmup 3 --eviction-requests 24 \
  --output /path/to/preview-cache.json
```

Repeat with the Dev checkpoint and a separate output file. Model paths must
refer to local checkpoint directories. Pin the model revision during download
and use the same image, code, settings and hardware for comparisons.

Each arm starts a fresh engine. Warmup uses distinct images of the same shape;
the measured target image is first introduced after warmup. The benchmark
compares first use, repeated images, changed text, changed processing options,
and distinct-image controls. Controls are deterministic shifted/pixel-modified
variants of the input with identical dimensions. If requested, additional
distinct images fill the encoder cache before reusing the original image.

`--eviction-requests` must be large enough for the actual visual-token capacity.
If the original feature remains resident, the benchmark fails the expected
re-encoding check; increase the count rather than interpreting that as a model
correctness failure. This test expects the CPU processor cache to remain large
enough to retain the original processed image while the GPU feature is evicted.

The JSON report records:

- Code commit/status, benchmark checksum, checkpoint config checksum, locally
  recorded Hub revision when available, dependency versions, GPU and config.
- Per-request wall time, generated token IDs/text, and finish reason.
- Processor cache hits and actual model `embed_multimodal` invocation/image
  counts, independent of timing improvements.
- SHA-256 fingerprints, shapes and dtypes of encoder outputs. Fingerprinting
  happens after each measured request, outside its timing interval.
- Unique encoder tensor storage bytes, GPU allocated/reserved/peak allocation,
  frontend RSS and processor-cache accounted size.
- Equality checks for generated tokens and encoder features between arms,
  expected hit/miss behavior, and median repeated/unique-image latency.

The benchmark exits unsuccessfully if an acceptance check fails. It writes the
completed disabled arm before starting the enabled arm, retaining partial
evidence if later startup fails. Instrumentation belongs only to the benchmark
worker extension; production models and cache policies are unchanged.

## Interpreting measurements

Latency is wall time around `Omni.generate`, including its request preprocessing,
queueing and generation. It excludes model startup, explicit warmup, image file
I/O, prompt construction and probe RPCs. The first encounter of a different
processing shape may still trigger JIT compilation and is reported separately.
Do not run other GPU jobs, downloads or CPU tests during performance collection.

The sender cache's accounted bytes represent retained receiver items, not
physical frontend tensor storage. Frontend RSS is whole-process memory, not a
measurement of cache memory alone. GPU peak allocation includes the model and
other allocations and is reset after warmup. Encoder storage bytes count unique
underlying storages, including any additional Dev visual features in them.

Seven requests are a small sample, and autoregressive decode can dominate the
end-to-end duration. Report raw timings and distinct-image controls alongside
any speedup; a hit does not guarantee a large end-to-end improvement.

## Measured results (2026-09-08)

Validation used checkout `c704aeccae192d7f88e9138271682ddad1327a69` with
the benchmark and tests in this change, one NVIDIA L40S (46068 MiB), driver
580.159.04, torch 2.13.0+cu130, vLLM 0.28.0, transformers 5.14.1, and
vllm-omni 0.28.1.dev102+gc704aecca. Checkpoint revisions were:

- Preview: `ef5a5e41dbf0de1ef6275586b7580f0d4248b4c6`.
- Dev: `461ad0d7d846bd5fa944619b08213a936eee2e30`.

Both runs used the command above, a screenshot resized to 448 x 361 RGB,
32 maximum output tokens, three warmup requests, seven repeats and 24 eviction
requests. Each variant passed all 210 GPU acceptance checks across 42 requests
per arm. Generated token IDs and complete encoder feature fingerprints matched
between cache-disabled and cache-enabled runs. Repeated images and changed text
hit the processor cache and encoded zero new images; changed images/options
encoded one image. After encoder eviction, the original image still hit the CPU
processor cache and correctly recomputed its visual features.

| Variant | Repeated, off (s) | Repeated, on (s) | Unique, off (s) | Unique, on (s) |
|---------|-----------------:|----------------:|---------------:|--------------:|
| Preview | 0.7533 | 0.7413 | 0.7556 | 0.7695 |
| Dev | 0.9567 | 0.9429 | 0.9535 | 0.9711 |

These are medians, with repeated-request reductions of approximately 1.6% and
1.4%. Seven samples do not establish statistical significance or a general
speedup. Unique-image controls were slightly slower with caching enabled.

The target encoder features were BF16 `[208, 3584]` for Preview (1.42 MiB)
and `[154, 16384]` for Dev (4.81 MiB, including deepstack features). At the first
repeat, total resident encoder tensor storage was 7.11/5.69 MiB (off/on) for
Preview and 24.06/19.25 MiB for Dev, including warmup entries. Enabled processor
cache accounting was 7.46 MiB and 7.22 MiB respectively. These values are not
whole-process GPU savings or physical frontend tensor memory.

The processor tests passed all nine cases for each checkpoint. Together with
the six existing model configuration cases, the final CPU runs passed 24 tests.
The GPU runs exercised the default LRU sender/receiver caches; shared-memory
cache mode and multi-worker configurations were not tested.

## Preview text-to-image regression

A separate single-L40S smoke test completed Preview AR-to-DiT generation with
cache enabled: seed 42, 256 x 256 output, five diffusion steps, and guidance 4.0.
The result was a nonblank RGB image. This checks pipeline execution, not image
quality or a text-to-image cache speedup.

At the validated checkout, the shared image example fails before inference
because it forwards diffusion defaults without a structured config owner for
this pipeline (including `cfg_parallel_size` and `enable_cpu_offload`). The smoke
test therefore used the minimal `Omni` API. To reproduce, copy the supplied
benchmark YAML, change `pipeline` to `mammoth_moda2`, and append this second stage:

```yaml
  - stage_id: 1
    devices: "0"
    max_num_seqs: 1
    gpu_memory_utilization: 0.16
    enforce_eager: true
    trust_remote_code: true
    enable_prefix_caching: false
```

Use that deployment file in this script, launched under a `__main__` guard:

```python
from copy import deepcopy

from diffusers.utils import numpy_to_pil
from PIL import Image

from vllm_omni import Omni
from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.model_extras.mammothmodal2_preview import build_text_to_image_prompt


def main():
    prompt = build_text_to_image_prompt(
        "A red ceramic mug on a white table, studio photograph.",
        None, height=256, width=256,
    )
    omni = Omni(
        model="/path/to/MammothModa2-Preview",
        deploy_config="/path/to/t2i.yaml",
        trust_remote_code=True,
    )
    try:
        params = deepcopy(omni.default_sampling_params_list)
        for item in params:
            item.seed = 42
            item.extra_args = {
                "text_guidance_scale": 4.0,
                "cfg_range": [0.0, 1.0],
                "num_inference_steps": 5,
            }
        params[0].max_tokens = 16 * 17 + 1
        params[0].temperature = 0
        outputs = list(omni.generate(prompt, sampling_params_list=params))
        result = extract_images_from_outputs(outputs)[0]
        result = result if isinstance(result, Image.Image) else numpy_to_pil(result)[0]
        assert result.size == (256, 256)
        result.save("mammoth-t2i-smoke.png")
    finally:
        omni.close()


if __name__ == "__main__":
    main()
```

## Scope and limitations

- Validation is single-worker, eager, BF16 AR-only with prefix caching disabled.
  It does not establish TP/PP, replicas, concurrent batching, quantization,
  compilation/CUDA graphs, ROCm, or image-editing support.
- Dev validation must include its full encoded feature representation, including
  the features consumed by deepstack; matching only final text is insufficient.
- `AsyncOmni.reset_encoder_cache()` is currently unsupported through the
  orchestrator. Use a fresh engine for a cold baseline. The processor tests
  exercise local clearing; this is not evidence of a complete serving reset API.
- Preview text-to-image is checked separately as described above. The shared
  example's argument compatibility issue remains unresolved. An AR image-cache benchmark does not validate DiT or
  claim improvements for a text-to-image request without an input image.
