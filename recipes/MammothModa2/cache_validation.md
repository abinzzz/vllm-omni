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

The pinned Dev checkpoint also declares `Qwen2VLImageProcessorFast` and
`Mammothmoda2Processor`, with `patch_size: 16`. Its Qwen3-VL encoder does not
imply that the outer image processor must be replaced. The processor tests
compare image tensors and grids against `AutoImageProcessor` loaded directly
from each checkpoint, in addition to comparing cache hits and misses.

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
  --mode validate \
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
- SHA-256 fingerprints, shapes and dtypes of the encoder outputs actually
  returned to embedding gather, in access order, as well as resident entries.
  A matching feature elsewhere in the cache cannot satisfy the equality check.
  Fingerprinting happens after each request.
- Unique encoder tensor storage bytes, GPU allocated/reserved/peak allocation,
  frontend RSS and processor-cache accounted size.
- Equality checks for generated tokens and encoder features between arms,
  finish reasons, expected hit/miss behavior, and text-only requests without
  image encoding or cache lookups.

The benchmark exits unsuccessfully if an acceptance check fails. It writes the
completed disabled arm before starting the enabled arm, retaining partial
evidence if later startup fails. Instrumentation belongs only to the benchmark
worker extension; production models and cache policies are unchanged.

Run performance measurements separately, without the worker extension or
per-request fingerprint RPCs:

```bash
python -m benchmarks.mammoth_moda2_cache \
  --model /path/to/MammothModa2-Preview \
  --deploy-config benchmarks/mammoth_moda2_cache.yaml \
  --image /path/to/image.png \
  --mode latency --repeats 30 --warmup 3 \
  --output /path/to/preview-latency.json
```

The latency mode still compares generated token IDs and finish reasons between
arms, but does not claim to verify encoder reuse; pair it with a successful
`validate` run. Reports retain every request latency and include sample count,
median, mean, standard deviation, minimum and maximum for repeated-image and
unique-image requests. Repeat complete runs to assess variation. If production
code changes, run the same harness and inputs on both pinned base/head checkouts.

## Interpreting measurements

Latency is wall time around `Omni.generate`, including its request preprocessing,
queueing and generation. It excludes model startup, explicit warmup, image file
I/O, prompt construction and probe RPCs. The first encounter of a different
processing shape may still trigger JIT compilation and is reported separately.
Do not run other GPU jobs, downloads or CPU tests during performance collection.
Only `--mode latency` produces timings without worker probes. Timings recorded
by `--mode validate` are diagnostic and should not be used to claim a speedup.

The sender cache's accounted bytes represent retained receiver items, not
physical frontend tensor storage. Frontend RSS is whole-process memory, not a
measurement of cache memory alone. GPU peak allocation includes the model and
other allocations and is reset after warmup. Encoder storage bytes count unique
underlying storages, including any additional Dev visual features in them.

Seven requests are a small sample, and autoregressive decode can dominate the
end-to-end duration. Report raw timings and distinct-image controls alongside
any speedup; a hit does not guarantee a large end-to-end improvement.

## Current validation (2026-09-11)

The current harness passed 36 CPU tests: ten benchmark verifier/probe tests,
six model configuration tests, and ten processor tests for each pinned
checkpoint below. The independent processor reference is converted to the
configured model dtype, matching vLLM's processor-output conversion, before
exact equality is checked for cache misses and hits.

Fresh validation used branch `40cbd81f8b07b4157f1a12973cffbc482da98614` and an
additional checkout of upstream `d3493384cfc6c4b9ba1fc4e534e91fe588ecb4f1`, each
with the validation changes applied. Both used the existing Python 3.12.14,
torch 2.13.0+cu129, vLLM 0.28.0+cu129, transformers 5.14.1, diffusers 0.40.0,
kernels 0.15.2 environment and driver 570.133.20. Main declares kernels 0.16.1;
these are scoped execution checks with the recorded runtime, not validation of
all current-main dependencies. The model implementations are identical at the
two code revisions; the main check also exercises its newer input renderer.

Each variant on each revision passed all **499 GPU acceptance checks**, across
83 measured requests per arm plus three distinct warmups. Runs used an L40S,
eager BF16, TP=1, one AR worker, 32 maximum output tokens, seven repeated images
and 64 eviction requests. The checked-in
`tests/assets/qwen_image_edit/qwen_image_edit_2511_test1.png` was resized to
262 x 448 RGB. This is a different input from the historical run below.

The checks establish actual preprocessing/encoder reuse, exact token and
consumed-feature parity, correct changed-image/option invalidation, text-only
behavior, and re-encoding after eviction. An independent comparison of the two
code revisions also matched request identity, tokens, finish reasons and
consumed features for all 166 requests per variant.

At the first repeated request, both revisions measured:

| Variant | Target feature | Resident encoder off/on (MiB) | Processor accounted on (MiB) | GPU peak off/on (MiB) |
| --- | --- | ---: | ---: | ---: |
| Preview | BF16 [144, 3584], 0.984 MiB | 4.922 / 3.938 | 5.168 | 35824.91 / 35824.91 |
| Dev | BF16 [112, 16384], 3.500 MiB | 17.500 / 14.000 | 5.250 | 35971.42 / 35971.42 |

Resident features include warmups; processor accounting is not physical frontend
RSS. Whole-engine peak allocation did not decrease in this configuration.

Three uninstrumented Preview latency rounds on `40cbd81f` used one dedicated
L40S, three warmups, 30 repeats and 30 distinct-image controls per arm. All
three rounds passed token and finish-reason equality checks. Values below are
per-round medians; JSON retains every sample and mean/stddev/min/max.

| Round | Repeated off/on (ms) | Unique off/on (ms) |
| --- | ---: | ---: |
| 1 | 534.287 / 527.328 | 620.062 / 634.734 |
| 2 | 535.171 / 526.578 | 612.705 / 642.181 |
| 3 | 537.090 / 526.937 | 618.009 / 636.032 |

Repeated-image reductions were 1.3-1.9%; unique-image controls were 2.4-4.8%
slower. Arms always ran off before on. The host had unrelated jobs on other
GPUs, although no other workload used the measured GPU. These results do not
establish statistical significance, throughput scaling or a general speedup.
They measure existing caching, not a production optimization introduced by
this change.

An independent three-round run on `d3493384`, on the same GPU and with the
same settings, also passed all token/finish-reason checks:

| Round | Repeated off/on (ms) | Unique off/on (ms) |
| --- | ---: | ---: |
| 1 | 537.462 / 528.210 | 619.885 / 636.750 |
| 2 | 536.113 / 523.861 | 622.369 / 622.136 |
| 3 | 539.432 / 508.661 | 623.165 / 615.856 |

The main run's repeated-image reductions range from 1.7% to 5.7%, while its
unique-image controls range from 2.7% slower to 1.2% faster. The third round
shifts both groups, reinforcing the shared-host and fixed-order limitations:
the full observed change cannot be attributed confidently to caching alone.

## Historical measured results (2026-09-08)

These results predate the separate latency mode and the consumed-feature
assertions. They describe the original instrumented benchmark, not a fresh run
of the current harness.

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
| --------- | -----------------: | ----------------: | ---------------: | --------------: |
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

An opt-in pytest regression runs both cache configurations on fresh engines,
at 256 x 256 / 5 steps and 512 x 512 / 50 steps, with both stages on one visible
L40S. It saves the effective deployment YAMLs,
PNG images and a JSON report under the pytest temporary directory, checks image
size and nonblank output, and compares all RGB pixels between cache off/on:

```bash
CUDA_VISIBLE_DEVICES=0 \
MAMMOTH_MODA2_T2I_MODEL=/path/to/MammothModa2-Preview \
  python -m pytest \
  tests/e2e/offline_inference/test_mammoth_moda2_cache_regression.py \
  -q -o addopts='' --basetemp=/path/to/t2i-regression-results
```

Use a new output directory for each run: pytest clears its `--basetemp` directory.
The regression uses seed 42 and guidance 4.0, with AR/DiT GPU memory fractions
0.8/0.16. The AR token limits are 273 and 1057 for the two resolutions. Both
configurations passed exact RGB equality on both code revisions above, and the
512px image also matched exactly across revisions. The 256px/5-step image is
visually poor and is only an execution smoke test; the 512px/50-step image
depicts the prompted red mug. This is not a dataset-level image-quality test.

The test validates the generation pipeline while changing the cache setting;
a text-only generation prompt does
not demonstrate an image-encoder cache hit or a DiT speedup. Model assets must
already be downloaded locally. Without `MAMMOTH_MODA2_T2I_MODEL`, collection
skips this GPU regression and does not download weights.

The historical 2026-09-08 single-L40S smoke test completed Preview AR-to-DiT generation with
cache enabled: seed 42, 256 x 256 output, five diffusion steps, and guidance 4.0.
The result was a nonblank RGB image. This checks pipeline execution, not image
quality or a text-to-image cache speedup.

At the historical checkout, the shared image example failed before inference
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
  example's historical argument issue is not a result of this cache validation.
  An AR image-cache benchmark does not validate DiT acceleration or claim
  improvements for a text-to-image request without an input image.
