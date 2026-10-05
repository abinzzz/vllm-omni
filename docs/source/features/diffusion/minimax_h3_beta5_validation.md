# MiniMax-H3 beta5 A1 validation report

Status: **A1 functional, HTTP task routing, fixed-seed comparison and media
checks complete.** The combined `--task-type auto` server mode remains unverified.
This report describes the tested host and workload; metrics do not establish
general quality superiority.

## Scope and pinned artifacts

- Repository: `https://github.com/abinzzz/vllm-omni.git`, branch `feat/h3-single-file-bf16`.
- beta5 source: `TenStrip/10Eros-Max`, revision `8a198588c8870ab0d613b3492a3150d091c8c2dd`.
- non-Turbo BF16 file: `10Eros_Max_h3_hybrid_beta5.safetensors`, SHA256 `a4c8fcefca0628f717ddfc91cadbce7bbef6a7da1fd470d144d81ef3ca4de33f`.
- Turbo BF16 file: `10Eros_Max_h3_TURBO-hybrid_beta5.safetensors`, SHA256 `098f138f3e899d03821dd8296c8db3db6fa16371e50fcdd38661d347fd9a1dfa`.
- Shared components: `MiniMaxAI/MiniMax-H3`, revision `42ed227ee7df40d41602854ae760620d6eb651fe`.
- Runtime: NVIDIA L40S x8 host, PyTorch `2.13.0+cu129`, vLLM `0.30.0+cu129`, vLLM-Omni `0.29.0rc2.dev46+g6753bbd18.d20260913`. The runtime warns that the Omni/vLLM minor versions differ.

The local safetensors file supplies the selected DiT. Tokenizer, processor,
Qwen3-VL encoder and video/audio VAEs are taken from the pinned MiniMax H3
component tree. The A1 loader preserves the single-file QKV row layout and
executes the compressed AdaLN curve/projection path. A1 accepts BF16/FP32
checkpoint tensors and explicitly rejects quantized layouts; INT8 and W4A8 are
outside this task. The merged Turbo deltas are already in the Turbo file; no
LightX2V adapter is used for beta5.

## Functional generation

Each beta5 variant generated T2VA, FL2VA and Ref2VA output at 1344×768, 107
frames and 32 kHz stereo audio. Non-Turbo used 50 Euler evaluations; Turbo
used 8 Euler evaluations. Both used guidance 1, video/audio flow shifts 12/3,
24 FPS and seed 42 in the single-run functional matrix. All six MP4s passed
full decode and H.264/AAC stream checks. The fixed-seed suite expanded to 36
commands (official and beta5 sides), using three seeds, one warmup and three
measured requests per side.

The independent media audit covered all 144 MP4s: command status, manifest
SHA256, expected `[107, 768, 1344, 3]` video arrays, `[2, 142400]` finite audio,
H.264 1344×768/107-frame metadata, AAC stereo 32 kHz metadata and full ffmpeg
decode all passed. Machine-local evidence is under:

- `$A1_VALIDATION_ROOT/evaluation-20261004-tp2-dlo-resident20/full-media-audit.json`
- `$A1_VALIDATION_ROOT/evaluation-20261004-tp2-dlo-resident20/generation/`

## Matched LPIPS, audio similarity and VBench

The suite compares official H3 with non-Turbo beta5. For Turbo, official H3
uses the pinned LightX2V 8-step adapter (`3ec17a324ced54151364f24f8b5fb6bf7e26414f`);
beta5 Turbo uses the merged checkpoint as stored. LPIPS is mean framewise
perceptual distance. CLAP values are left/right/mono audio embedding cosine
similarities. VBench lists subject consistency, background consistency, motion
smoothness and aesthetic quality; these are four selected single-video metrics,
not a full-dataset VBench aggregate.

## LPIPS and audio similarity

| Case | Mean LPIPS | CLAP L / R / mono |
|---|---:|---|
| 00-non-turbo-t2va-seed42 | 0.268509 | 0.181871 / 0.172217 / 0.155214 |
| 00-turbo-t2va-seed42 | 0.346749 | 0.368019 / 0.368447 / 0.351151 |
| 00-non-turbo-fl2va-seed42 | 0.031431 | 0.986571 / 0.995152 / 0.982536 |
| 00-turbo-fl2va-seed42 | 0.180905 | 0.974304 / 0.963358 / 0.985811 |
| 00-non-turbo-ref2va-seed42 | 0.490968 | 0.303977 / 0.328659 / 0.334715 |
| 00-turbo-ref2va-seed42 | 0.418393 | 0.112248 / -0.049186 / -0.038390 |
| 01-non-turbo-t2va-seed2026 | 0.626634 | 0.651987 / 0.635885 / 0.644873 |
| 01-turbo-t2va-seed2026 | 0.621969 | 0.383208 / 0.351678 / 0.382042 |
| 01-non-turbo-fl2va-seed2026 | 0.159896 | 0.098102 / 0.098145 / 0.115039 |
| 01-turbo-fl2va-seed2026 | 0.653602 | 0.559143 / 0.589905 / 0.588219 |
| 01-non-turbo-ref2va-seed2026 | 0.669923 | 0.627448 / 0.584115 / 0.673286 |
| 01-turbo-ref2va-seed2026 | 0.640217 | 0.327105 / 0.322203 / 0.306022 |
| 02-non-turbo-t2va-seed7 | 0.416382 | 0.891937 / 0.884236 / 0.832317 |
| 02-turbo-t2va-seed7 | 0.393279 | 0.593052 / 0.631086 / 0.645776 |
| 02-non-turbo-fl2va-seed7 | 0.018618 | 0.993620 / 0.994144 / 0.987819 |
| 02-turbo-fl2va-seed7 | 0.029846 | -0.011210 / 0.895061 / 0.136745 |
| 02-non-turbo-ref2va-seed7 | 0.633482 | 0.257819 / 0.228550 / 0.223561 |
| 02-turbo-ref2va-seed7 | 0.139015 | 0.218321 / 0.199069 / 0.235836 |

## VBench (subject / background / motion / aesthetic)

| Case | Official S / B / M / A | beta5 S / B / M / A |
|---|---:|---:|
| 00-non-turbo-t2va-seed42 | 0.979290 / 0.970118 / 0.996458 / 0.410803 | 0.954450 / 0.888356 / 0.996437 / 0.436920 |
| 00-turbo-t2va-seed42 | 0.965078 / 0.922041 / 0.995080 / 0.432220 | 0.927479 / 0.907300 / 0.994563 / 0.393799 |
| 00-non-turbo-fl2va-seed42 | 0.989929 / 0.955638 / 0.996191 / 0.403038 | 0.980496 / 0.931910 / 0.995734 / 0.399607 |
| 00-turbo-fl2va-seed42 | 0.991156 / 0.953503 / 0.996254 / 0.406310 | 0.750115 / 0.898894 / 0.995490 / 0.496247 |
| 00-non-turbo-ref2va-seed42 | 0.986539 / 0.915147 / 0.997057 / 0.494805 | 0.995113 / 0.964814 / 0.996691 / 0.497898 |
| 00-turbo-ref2va-seed42 | 0.781782 / 0.918287 / 0.996923 / 0.466086 | 0.983708 / 0.949246 / 0.996564 / 0.481310 |
| 01-non-turbo-t2va-seed2026 | 0.942472 / 0.935948 / 0.992405 / 0.602400 | 0.915135 / 0.947457 / 0.994385 / 0.556092 |
| 01-turbo-t2va-seed2026 | 0.936817 / 0.884816 / 0.989251 / 0.548974 | 0.933998 / 0.933951 / 0.986425 / 0.611727 |
| 01-non-turbo-fl2va-seed2026 | 0.974931 / 0.948256 / 0.996934 / 0.479502 | 0.856371 / 0.927750 / 0.996215 / 0.518525 |
| 01-turbo-fl2va-seed2026 | 0.964359 / 0.953014 / 0.996768 / 0.496605 | 0.677221 / 0.854057 / 0.996268 / 0.571536 |
| 01-non-turbo-ref2va-seed2026 | 0.953616 / 0.942344 / 0.997235 / 0.543671 | 0.938668 / 0.952227 / 0.995352 / 0.530809 |
| 01-turbo-ref2va-seed2026 | 0.963277 / 0.961083 / 0.997031 / 0.573784 | 0.933160 / 0.962453 / 0.992599 / 0.528553 |
| 02-non-turbo-t2va-seed7 | 0.970275 / 0.964056 / 0.996053 / 0.394368 | 0.836879 / 0.957096 / 0.996119 / 0.392017 |
| 02-turbo-t2va-seed7 | 0.961412 / 0.964367 / 0.994408 / 0.380029 | 0.913038 / 0.976887 / 0.995046 / 0.409885 |
| 02-non-turbo-fl2va-seed7 | 0.976621 / 0.980545 / 0.997124 / 0.303622 | 0.962983 / 0.970924 / 0.996902 / 0.299319 |
| 02-turbo-fl2va-seed7 | 0.923639 / 0.962649 / 0.996742 / 0.340295 | 0.980721 / 0.964028 / 0.996519 / 0.287256 |
| 02-non-turbo-ref2va-seed7 | 0.823779 / 0.949737 / 0.996608 / 0.475867 | 0.987480 / 0.977979 / 0.996936 / 0.446937 |
| 02-turbo-ref2va-seed7 | 0.879780 / 0.942516 / 0.996979 / 0.394138 | 0.966329 / 0.955039 / 0.996933 / 0.423654 |

## Request timing

Timing is the median of three measured requests after one warmup; it includes
generation, MP4 mux and output write, and excludes model startup and
verification. Values are seconds per request.

| Matched case | Official H3 | beta5 |
|---|---:|---:|
| 00-non-turbo-fl2va-seed42 | 519.71 | 517.93 |
| 00-non-turbo-ref2va-seed42 | 481.75 | 480.57 |
| 00-non-turbo-t2va-seed42 | 439.70 | 437.63 |
| 00-turbo-fl2va-seed42 | 98.76 | 94.85 |
| 00-turbo-ref2va-seed42 | 92.37 | 88.01 |
| 00-turbo-t2va-seed42 | 84.77 | 80.63 |
| 01-non-turbo-fl2va-seed2026 | 518.66 | 518.10 |
| 01-non-turbo-ref2va-seed2026 | 479.96 | 477.58 |
| 01-non-turbo-t2va-seed2026 | 440.22 | 439.49 |
| 01-turbo-fl2va-seed2026 | 98.42 | 94.67 |
| 01-turbo-ref2va-seed2026 | 91.86 | 87.53 |
| 01-turbo-t2va-seed2026 | 85.30 | 80.60 |
| 02-non-turbo-fl2va-seed7 | 518.32 | 518.00 |
| 02-non-turbo-ref2va-seed7 | 480.60 | 477.97 |
| 02-non-turbo-t2va-seed7 | 439.48 | 438.35 |
| 02-turbo-fl2va-seed7 | 98.52 | 94.35 |
| 02-turbo-ref2va-seed7 | 92.37 | 87.70 |
| 02-turbo-t2va-seed7 | 85.35 | 80.15 |

Scores and timings describe only these matched seeds/prompts. LPIPS measures
difference from the reference; it does not establish which output is better.
CLAP cosine similarity is not a direct audio quality or synchronization score.
No numeric quality threshold was specified in the acceptance criteria.

## Validated deployment profiles

| Profile | Coverage | Result |
|---|---|---|
| HSDP4: TP1, Ulysses 4, text-encoder TP4, VAE patch parallel 4 | Both variants × T2VA/FL2VA/Ref2VA, formal 1344×768 | Six functional cases passed |
| TP2 + distributed layerwise offload, resident layers 20 | Both variants × all tasks; 18 matched fixed-seed official/beta5 pairs | Quality and timing suite passed |
| TP2 + DLO, resident layers 0 | Both variants × all tasks; one full-resolution request per case | Six isolated functional cases passed |
| TP4 | Turbo × all tasks | Three functional cases passed |

Not established: non-Turbo TP4, pre-sharded HSDP (explicitly rejected because
H3 tensor transforms cannot be applied by that shared loader), other GPU counts
or replication profiles, CPU-offload and mixed offload combinations, and
concurrent independent TP2+DLO requests. DLO runs completed but emitted the
generic Orchestrator 30-second shutdown timeout and shared-memory
resource-tracker warning. One concurrent resident-zero DLO FL2VA trial timed
out; its isolated retry passed. The cause of the concurrent failure is not
established. Detailed run manifests and per-profile evidence are under
`$A1_VALIDATION_ROOT/evaluation-20261004-tp2-dlo-resident20/` and
`$A1_VALIDATION_ROOT/PROFILE_MATRIX.md`.

## HTTP server status

The task-specific `vllm serve` entrypoint was verified with the beta5 non-Turbo
file, pinned base revision and HSDP4. The `fl2va` server returned T2VA and
first/last-keyframe FL2VA MP4s through `/v1/videos/sync`; a separate `ref2va`
server returned an image-reference MP4. All three were 107-frame H.264 at
448×256 with 32 kHz stereo AAC and passed full ffmpeg decode. The smoke requests
used two Euler steps and establish API routing/media output, not quality. MP4,
ffprobe JSON and SHA256 evidence are under
`$A1_VALIDATION_ROOT/http-smoke-20261005/`.

Use `--model-class-name MiniMaxH3Pipeline` when passing a local single-file
checkpoint, and ensure the launched vLLM-Omni code comes from this checkout
(`PYTHONPATH="$PWD"` in a source checkout). A combined `--task-type auto`
startup attempt was killed at rank 2 before endpoint readiness; the cause was
not established. Keep FL2VA/T2VA and Ref2VA as separate task-specific services,
as shown in the README and the passing checks above.

## Reproduction and evidence

The host-only reproducibility bundle is kept outside the repository at
`/home/huxiaobin/chenyb/validation/h3-a1/` (set `A1_VALIDATION_ROOT` to this
path) so benchmark
outputs and large videos are not included in the source change. Each `run-case.py`
manifest records the complete command, model/base revisions, SHA256 hashes,
seed, sampler, inputs, source checkout paths, runtime versions, frames/audio
shapes, request timings and ffprobe result. A representative case command is:

```bash
export A1_VALIDATION_ROOT=/home/huxiaobin/chenyb/validation/h3-a1
export BASE_ROOT=/path/to/local/MiniMax-H3-pinned-snapshot
export CHECKPOINT=/path/to/10Eros_Max_h3_hybrid_beta5.safetensors
export OUT="$A1_VALIDATION_ROOT/reproduction/non-turbo-t2va-seed42"

PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0,1 \
  /home/huxiaobin/chenyb/.venvs/vllm-029-validation/bin/python \
  "$A1_VALIDATION_ROOT/run-case.py" \
  --model "$CHECKPOINT" --base-root "$BASE_ROOT" \
  --base-revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --task t2va --profile tp2-dlo --resident-layers 20 \
  --steps 50 --seed 42 --width 1344 --height 768 --duration 4 \
  --flow-shift 12 --audio-flow-shift 3 \
  --prompt 'A red ceramic teapot on a wooden table, steam rising gently, with quiet room ambience.' \
  --warmups 1 --repeats 3 --output-dir "$OUT"
```

For FL2VA, add `--image /path/to/first.png --image /path/to/last.png`; for
Ref2VA, add one or more `--image` options. For Turbo, use the Turbo file and
`--steps 8`. The fixed-seed evaluation plan and input fixtures are in
`$A1_VALIDATION_ROOT/evaluation-20261004-tp2-dlo-resident20/`. Re-run the
media audit with:

```bash
/home/huxiaobin/chenyb/.venvs/vllm-029-validation/bin/python \
  "$A1_VALIDATION_ROOT/audit-fixed-seed-media.py" \
  "$A1_VALIDATION_ROOT/evaluation-20261004-tp2-dlo-resident20"
```

Focused regressions passed 314 tests with 3 skipped, and the synthetic H3
parallel/offload suite passed 62 tests. Re-run from the repository root with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  /home/huxiaobin/chenyb/.venvs/vllm-029-validation/bin/python -m pytest -q \
  tests/diffusion/models/minimax_h3/test_single_file.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_modulation.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_contract.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_packing.py
```

Full logs, fixed inputs, prompts, metric JSON, side-by-side videos and
per-request manifests remain at `$A1_VALIDATION_ROOT`. This report includes
summary measurements for review; external raw evidence is host-specific and
must be regenerated if that host or cache is unavailable.
