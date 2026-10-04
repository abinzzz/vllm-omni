<!-- BEGIN H3 A1 BRANCH TODO -->

# MiniMax-H3 A1: BF16 single-file checkpoint support

This branch tracks implementation and validation of TenStrip/10Eros-Max beta5
BF16 Turbo and non-Turbo checkpoints in vLLM-Omni. Support is **in development;
full acceptance validation is not complete yet**. Keep this checklist current as work progresses; mark an item
complete only when its implementation or validation evidence is recorded.

Branch: `feat/h3-single-file-bf16`

Initial upstream base: `ee8fdab1dc26ea8f09d7910cdc98a17c4b0ea02a`

The upstream project README is preserved below. Once A1 acceptance is complete,
replace this temporary section with verified user-facing documentation and a
link to the serving recipe, retaining the general project README. INT8 and W4A8
support belongs to A2 and is outside this checklist.

## Acceptance criteria

- Load either beta5 BF16 checkpoint by pointing at one file, obtaining remaining
  components from the base H3 repository.
- Execute the checkpoint as stored, including its compressed AdaLN computation.
- Generate valid MP4s with audio for T2VA, FL2VA and Ref2VA with both variants.
- Compare against official H3 at fixed seeds using LPIPS, audio similarity,
  VBench and seconds per request. The Turbo reference uses LightX2V Turbo.
- Validate supported official-H3 parallelism and offload profiles, and report
  any coverage gaps explicitly.

## TODO

### 0. Workspace and baseline

- [x] Create an isolated worktree and A1 branch from the fetched upstream main.
- [x] Verify GPU discovery outside the sandbox: 8 L40S GPUs; PyTorch reports CUDA available.
- [x] Run the existing modulation CPU baseline: 6 passed, 4 deselected.
- [x] Establish a compatible complete inference environment, including compiled vLLM extensions.
- [x] Inventory checkpoint storage requirements and available GPUs before downloads or inference.

The CPU baseline only checks existing behavior; it is not evidence of A1 support.
Machine-specific logs and environment notes live outside the repository in
`~/chenyb/validation/h3-a1/`, including `HANDOFF.md` and `baseline-modulation.log`.
GPU occupancy must be checked again before each run.

The validated host environment imports this worktree, reports CUDA available
with eight devices, and loads the compiled `vllm._C_stable_libtorch` extension.
It uses vLLM-Omni `0.29.0rc2.dev46` with vLLM `0.30.0+cu129`; Omni emits a
minor-version mismatch warning. All six H3 functional cases and the focused
HSDP/offload regressions ran in this environment. This evidence is specific to
the recorded host profile, not a general compatibility claim.

### 1. Checkpoint and reference investigation

- [x] Pin beta5 BF16 Turbo/non-Turbo filenames, revisions and checksums, and base H3 component revisions.
- [x] Inspect checkpoint metadata, tensor names, dtypes and shapes before large downloads where possible.
- [x] Identify which components are stored in the file and which must come from base H3.
- [x] Read the reference implementation and document compressed AdaLN computation and parameter mapping.
- [x] Verify both checkpoints in T2VA, FL2VA and Ref2VA; inspect the controlled FL2VA motion sample.
- [x] Record the Turbo step count, native H3 shifted-sigma settings, guidance and baked-in deltas.
- [x] Finalize the loading interface and compatibility design from this evidence.

The pinned base snapshot's FL2VA audio and video VAE payloads have SHA256
`37dddc2f…dade5ea2` and `5f0c2e16…befe0d3`. The official Ref2VA audio and
video VAE entries report the same full hashes ([audio](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/ce18b0e8462a05d7dbf960e3e3a3517570f20b07/Ref2VA/audio_vae/model.safetensors),
[video](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/c4ccdbc27aed6da9462aae7de67d313e47d055b4/Ref2VA/video_vae/source/model.safetensors)).
The first Ref2VA text-encoder shard also matches the pinned shared encoder at
SHA256 `6b9dfbc9…cca27cb` ([Ref2VA shard](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/Ref2VA/text_encoder/model-00001-of-00014.safetensors)).
This supports reusing the FL2VA base component tree for Ref2VA; full pinned
FL2VA payload checksums are recorded in
`~/chenyb/validation/h3-a1/base-payload-verification.json`.

### 2. Loader and model implementation

- [x] Reuse the native single-file entrypoint and existing component loading mechanisms where applicable.
- [x] Register native H3 single-file configuration and preserve multimodal/reference-input metadata.
- [x] Resolve the base repository separately from the checkpoint and load only the required base components.
- [x] Construct and execute the compressed AdaLN layout without expanding it into a full dense projection.
- [x] Implement FP32 curve interpolation and compressed projection primitives, with endpoint and dense-path regression tests.
- [x] Select curve time embeddings in the native DiT, load the stored table, and check FP32 invariants after host-weight restore.
- [x] Map and validate checkpoint weights, rejecting unsupported layouts and unexpected missing parameters.
- [x] Validate the complete compressed single-file tensor schema and derive DiT configuration from file descriptors.
- [x] Reject unsupported A2 quantized files with actionable errors.
- [x] Ensure the selected checkpoint supplies the DiT weights for every task; prevent silent base-DiT fallback.
- [x] Preserve compressed AdaLN parameter dtypes in the HSDP precision policy; a complete small DiT forward matches unsharded execution on two GPUs.
- [x] Bind the selected single file and curve table in host-weight plans, with FP32 restoration matching ordinary loading.
- [x] Run ordinary HSDP checkpoint loading and cold/warm video/audio forwards on two GPUs with a synthetic compressed checkpoint.
- [x] Keep the official grouped-checkpoint loader path intact while integrating the single-file path with AdaLN caching, offload and parallel loading.
- [x] Run Turbo with a documented author-recommended sampler/step setting without reapplying merged deltas.

The native grouped-checkpoint path retains its existing transformer source,
QKV row reordering and AdaLN sidecar eligibility; the single-file logic is
gated on a regular local file. A synthetic CUDA regression on two L40S GPUs
passed the dense and compressed-AdaLN runtime-projection-cache cases at TP1
and TP2, including warm-cache parity and rebuilding the collective after one
rank invalidates its local cache. The compressed cache test did not load H3
weights. Beta5 itself has successful HSDP4 functional runs and one small TP2 +
DLO T2VA smoke, while a synthetic HSDP two-GPU load/forward test passed. These
checks cover the code-path integration; they do not establish official H3
checkpoint loading or complete the production offload/parallelism matrix. The
remaining real-checkpoint profile matrix is tracked separately in
`~/chenyb/validation/h3-a1/PROFILE_MATRIX.md`.

Pre-sharded HSDP loading currently rejects H3 checkpoints because the shared loader
requires runtime-layout weights and cannot apply H3 tensor transforms. Ordinary
HSDP loading is tested above; production checkpoint profiles still require validation.

### 3. Focused regression tests

- [x] Test weight mapping and base-component selection with small synthetic checkpoints.
- [x] Test malformed, incomplete and unsupported checkpoints and clear error reporting.
- [x] Compare compressed AdaLN outputs with the reference computation on small tensors, documenting tolerances.
- [x] Cover task/partition selection and record the tested Turbo schedule.
- [x] Run focused H3, single-file loader, configuration and two-GPU HSDP regressions.
- [x] Run repository formatting and lint checks for changed files.

The latest host-side focused H3 regression covered single-file schema/dispatch,
AdaLN modulation, H3 contracts and packed conditioning: 314 passed and 3 were
skipped. It includes default-base revision, caller override, custom-base, and
local-base selection cases. The log is in
`~/chenyb/validation/h3-a1/current-turn-base-pin-regression.log`.
An additional synthetic parallel/offload regression passed 62 tests across
`test_minimax_h3_parallel.py`, `test_minimax_h3_offload.py`,
`test_minimax_h3_dlo_lifecycle.py`, and
`test_minimax_h3_vae_split_residency.py`. It used CUDA device 0 only because
the split-residency fixtures create CUDA streams; it did not load checkpoint
weights. Reproduce from the repository root with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  /home/huxiaobin/chenyb/.venvs/vllm-029-validation/bin/python -m pytest -q \
  tests/diffusion/models/minimax_h3/test_minimax_h3_parallel.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_offload.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_dlo_lifecycle.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_vae_split_residency.py
```

The full log is `~/chenyb/validation/h3-a1/current-turn-parallel-offload-regression.log`.
This synthetic coverage does not close the real-checkpoint deployment profile
matrix above.
An initial in-sandbox run could not allocate localhost sockets and left platform
fixtures uninitialized; it is not counted as a code failure.
A synthetic two-GPU HSDP test also passed on CUDA 0/1, covering selected-file
loading, FP32 AdaLN restoration and explicit rejection of pre-sharded loading;
it does not load a real beta5 checkpoint. See
`~/chenyb/validation/h3-a1/current-single-file-hsdp-device.log`.

### 4. End-to-end generation

Start with small-shape smoke tests, then repeat at the recorded acceptance shapes.
Complete each row only after preserving the command, configuration, seed, logs
and output evidence.

| Variant | T2VA | FL2VA | Ref2VA |
| --- | --- | --- | --- |
| beta5 BF16 non-Turbo | 50-step HSDP4 functional case passed; quality A/B pending | 50-step MP4/audio generated; seed-42 toy motion observed; quality A/B pending | 50-step HSDP4 case passed; quality A/B pending |
| beta5 BF16 Turbo | 8-step HSDP4 functional case passed; LightX2V A/B pending | 8-step MP4/audio generated; seed-42 toy motion observed; quality A/B pending | 8-step HSDP4 case passed; quality A/B pending |

Both files completed all three tasks using their selected single-file DiT weights.
Formal cases use seed 42, 1344×768 and 107 video frames with 32 kHz stereo
audio. Non-Turbo uses 50 Euler evaluations; Turbo uses Euler/simple at 8 steps,
within the author's recommended 4–8 range. Turbo deltas are already baked into
the file, so no LightX2V adapter is applied to it. Both variants use the native
H3 shifted sigma ladder with video/audio flow shifts 12/3 and guidance scale 1.
These settings establish runnable task paths; they do not settle relative
quality. In the seed-42 toy scene, both variants show the red ball moving from
the first-frame position toward the last-frame position. This is one controlled
sample, not a general motion-quality claim or an official-H3 comparison.

A source-path audit found no dropped-condition handoff: two FL2VA images default
to frame indices `[0, -1]`, are VAE-encoded, enter packed non-update condition
rows, and are reset from the condition anchor at every denoising step. The
focused keyframe-index and packed-row regressions pass (4 cases). A separate
seed-42 diagnostic tracks the red-object centroid at frames 0/26/53/80/106:
non-Turbo moves from x=240.1 to 999.9 pixels, and Turbo from x=240.2 to 1000.1,
with monotonic intermediate positions. This confirms motion in this toy sample
only; paired quality acceptance remains pending. Evidence:
`~/chenyb/validation/h3-a1/fl2va-motion-audit-seed42.json`. Reproduce with
`/home/huxiaobin/chenyb/.venvs/vllm-029-validation/bin/python ~/chenyb/validation/h3-a1/audit-fl2va-motion.py`.

#### Serving a beta5 file

This branch accepts a local beta5 BF16 safetensors file as the model argument;
the file supplies the DiT while the selected base H3 revision supplies the
tokenizer, text encoder, and audio/video VAEs. Review the checkpoint and base
model licenses above before downloading or serving them. Use a separate server
for each beta5 file. The following HSDP4 configuration matches the tested
1344×768 generation profile; the single-file pipeline and task routing were
verified through `Omni.generate`. The real Omni CLI parser and validation also
accept this command's HSDP4 options and base-revision JSON with a placeholder
file; no model is loaded in that check. An HTTP `/v1/videos` server request has
not yet been run on this host.

When `base_model` is omitted from `custom_pipeline_args`, the loader defaults to
`MiniMaxAI/MiniMax-H3` at the pinned revision
`42ed227ee7df40d41602854ae760620d6eb651fe`. An explicit `base_revision` remains
in effect, and a custom `base_model` is not silently assigned the MiniMax
revision.

```bash
export CHECKPOINT=/path/to/10Eros_Max_h3_hybrid_beta5.safetensors
export BASE_MODEL=MiniMaxAI/MiniMax-H3
export BASE_REVISION=42ed227ee7df40d41602854ae760620d6eb651fe
export PORT=8091

CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve "${CHECKPOINT}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type auto \
  --custom-pipeline-args "{\"base_model\":\"${BASE_MODEL}\",\"base_revision\":\"${BASE_REVISION}\"}" \
  --num-gpus 4 \
  --tensor-parallel-size 1 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --usp 4 \
  --ring 1 \
  --use-hsdp \
  --hsdp-shard-size 4 \
  --hsdp-replicate-size 1 \
  --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

For beta5 Turbo, set `CHECKPOINT` to
`10Eros_Max_h3_TURBO-hybrid_beta5.safetensors`. The merged Turbo file already
contains its deltas: use 8 Euler steps, guidance 1, and video/audio shifts
12/3; do not attach the LightX2V adapter. For non-Turbo, the validated
acceptance-shape setting is 50 Euler steps with the same guidance and shifts.
The server's `/v1/videos/sync` request chooses `t2va`, `fl2va`, or `ref2va` in
`extra_params.task`; see the [H3 recipe's HTTP examples](recipes/MiniMaxAI/MiniMax-H3.md#http-api-examples)
for request forms and media-input fields. The selected checkpoint and base
component revision are recorded in each generation manifest outside this repo.

With `PYTHONPATH` pointed at this worktree, non-Turbo T2VA produced a fully
decoded 448×256 MP4 with 107 frames and 32 kHz stereo audio at two Euler
evaluations on two L40S GPUs using TP2 and rank-local DLO. Request time including
mux/write was 9.00 s, excluding 100.11 s startup. This verifies loading and
media output only, not acceptance quality.

A pre-fix 50-step HSDP4 run at the RFC's 1344×768 shape completed with valid
media but produced only gray-brown texture. It used the A1 worktree, but its
native loader incorrectly applied grouped-QKV row reordering to ComfyUI-layout
single-file weights. Treat this run as a preserved pre-fix diagnostic, not as
current behavior. Its manifest, log and video are under
`~/chenyb/validation/h3-a1/generation/non-turbo-t2va-hsdp4-50step-branch/`.

The clip-length check requested 5 seconds with the same seed, prompt, checkpoint,
steps and resolution. H3 aligned this request to 124 frames; that MP4 also decodes
fully with stereo audio, but its middle frame has the same texture-only failure.
This result predates the QKV correction below and does not isolate clip length.
Startup took 173.11 s and generation plus mux/write took 323.54 s on HSDP4. Evidence is under
`~/chenyb/validation/h3-a1/generation/non-turbo-t2va-hsdp4-50step-duration5/`.

A one-step tensor trace found the first large activation in the base Qwen3-VL
text encoder's layer-6 MLP: the first prompt token reaches about 14k in hidden
dimension 731. The same prompt and base checkpoint through the official
Transformers Qwen3-VL implementation produce the same layer-6 and layer-50
outlier (within BF16 rounding), so this is expected reference behavior.

The texture failure was caused by QKV layout handling. ComfyUI's H3 attention
uses contiguous `[all Q, all K, all V]` projection rows, while the native
checkpoint path previously reordered them as grouped `[q0,k0,v0,q1,k1,v1,…]`.
The single-file path now preserves stored row order; native grouped checkpoints
retain their existing reorder. A regression test caught the incorrect order
before the fix. Commit `e310edd9` fixes it. On that commit, the same beta5
non-Turbo checkpoint generated the prompted moving ball/table scene in both a
448×256, 50-step, 5-second diagnostic and a 1344×768, 50-step, 4-second
HSDP4 functional run. The formal-shape run took 163.77 s to start and 269.83 s
for generation, mux and write; its MP4 fully decodes to 107 frames with 32 kHz
stereo audio. Evidence is under
`~/chenyb/validation/h3-a1/generation/non-turbo-t2va-qkv-direct-hsdp4-50step-1344x768-retry1/`;
the contact sheet is
`~/chenyb/validation/h3-a1/frames/non-turbo-qkv-direct-1344x768-contact.png`.
These are functional checks for one non-Turbo T2VA input, not the required
fixed-seed quality A/B or full A1 acceptance. Temporary instrumentation has
been removed; the pre-fix and fixed-run manifests and logs are retained under
`~/chenyb/validation/h3-a1/`.

Non-Turbo FL2VA and Ref2VA conditioning smokes also passed on commit
`6c0021ac`. Both used seed 42, 448×256, 2 steps and HSDP4. FL2VA consumed the
suite's first and last synthetic frames; Ref2VA consumed its first frame. Each
produced a fully decoded 107-frame MP4 with 32 kHz stereo audio; generation,
mux and write took 9.69 s and 9.30 s respectively. These small-step cases
verify request routing and media output only. Manifests, inputs, logs and videos
are under `~/chenyb/validation/h3-a1/generation/` and
`~/chenyb/validation/h3-a1/inputs/`; formal-shape runs and quality comparisons
remain pending.

A formal-shape non-Turbo FL2VA case completed on commit `0ff46502` using the
same first/last images, seed 42, 50 steps and 1344×768 HSDP4. Startup took
167.22 s; generation, mux and write took 313.47 s. Its MP4 fully decodes to
107 frames with 32 kHz stereo audio. The endpoints are included as conditions,
and sampled frames show motion between those endpoints. A later five-frame
centroid audit confirms left-to-right movement in this seed-42 toy clip. Treat
this as functional motion evidence for one input, not as FL2VA quality
acceptance.
Evidence is under
`~/chenyb/validation/h3-a1/generation/non-turbo-fl2va-qkv-direct-hsdp4-50step-1344x768/`;
the sampled frames are in
`~/chenyb/validation/h3-a1/frames/non-turbo-fl2va-qkv-direct-1344x768-contact.png`.

A formal-shape non-Turbo Ref2VA case also completed at 50 steps, seed 42 and
1344×768 HSDP4 using the suite's first image. Startup took 164.29 s and
generation, mux and write took 290.99 s. The MP4 fully decodes to 107 frames
with 32 kHz stereo audio. Sampled frames retain the reference scene; this is a
single-case media and conditioning check, not the required fixed-seed official
H3 A/B. Evidence is under
`~/chenyb/validation/h3-a1/generation/non-turbo-ref2va-qkv-direct-hsdp4-50step-1344x768/`;
the contact sheet is
`~/chenyb/validation/h3-a1/frames/non-turbo-ref2va-qkv-direct-1344x768-contact.png`.

The beta5 Turbo T2VA checkpoint also completed an 8-step HSDP4 case at seed 42
and 1344×768. The author recommends 6–8 steps for Turbo and the prepared suite
uses 8. Startup took 162.50 s; generation, mux and write took 51.20 s. Its MP4
fully decodes to 107 frames with 32 kHz stereo audio; sampled frames show the
prompted ball/table scene. The checkpoint SHA256 is
`098f138f3e899d03821dd8296c8db3db6fa16371e50fcdd38661d347fd9a1dfa`.
Evidence is under
`~/chenyb/validation/h3-a1/generation/turbo-t2va-qkv-direct-hsdp4-8step-1344x768/`.
This is a functional check only; comparison against official H3 with the
LightX2V Turbo adapter and quality metrics remains pending.

Turbo FL2VA also completed at seed 42, 8 steps, 1344×768 and HSDP4 using the
same first/last images. Startup took 164.14 s; generation, mux and write took
59.29 s. Its MP4 fully decodes with 107 frames and 32 kHz stereo audio. The
first and last output frames preserve their corresponding input images. A later
five-frame centroid audit confirms left-to-right motion in this seed-42 toy
clip; it is not FL2VA quality acceptance. Evidence is under
`~/chenyb/validation/h3-a1/generation/turbo-fl2va-qkv-direct-hsdp4-8step-1344x768/`;
the sampled frames are in
`~/chenyb/validation/h3-a1/frames/turbo-fl2va-qkv-direct-1344x768-contact.png`.

Turbo Ref2VA completed at seed 42, 8 steps and 1344×768 HSDP4 using the suite's
first image. Startup took 179.57 s; generation, mux and write took 55.28 s. The
MP4 fully decodes with 107 frames and 32 kHz stereo audio; sampled frames retain
the reference scene and show the ball moving across the shot. Evidence is under
`~/chenyb/validation/h3-a1/generation/turbo-ref2va-qkv-direct-hsdp4-8step-1344x768/`;
the contact sheet is
`~/chenyb/validation/h3-a1/frames/turbo-ref2va-qkv-direct-1344x768-contact.png`.
This is functional evidence for one reference case; official H3/LightX2V quality
comparisons and broader input coverage remain pending.

An earlier acceptance-shape run produced three 1344×768, 50-step MP4s, but the
runner imported vLLM-Omni from the separate Mammoth editable checkout instead
of this branch. Those outputs and later TP4/HSDP4 initialization failures are
preserved for diagnosis but are invalid as A1 implementation evidence. All
generation must set `PYTHONPATH` to this worktree and record the imported
module paths before it counts toward acceptance. See the external handoff and
logs under `~/chenyb/validation/h3-a1/` for details.

- [x] Generate all six formal-shape cases above; confirm seed-42 toy-scene FL2VA motion in both variants.
- [x] Verify MP4 decoding, dimensions, frame count, duration and audio track for every case.
- [x] Verify that first/last-frame and reference conditioning enter the intended inference path.
- [x] Verify actual checkpoint weight consumption for each task from the selected-file runtime manifests.
- [ ] Validate the parallelism and offload profile matrix and record unsupported or untested combinations.

Current evidence covers all six BF16 task/variant cases with HSDP4, a
synthetic compressed-AdaLN TP4 forward, and full-resolution T2VA with TP2 plus
distributed layerwise offload (DLO) for both checkpoints. Non-Turbo used 50
steps and completed one warmup plus three measured requests at 437.64, 523.73
and 436.85 s; the 523.73 s request overlapped with the Turbo DLO run. Turbo
used 8 steps and completed one warmup plus three measured requests at 80.56,
82.39 and 80.45 s. All outputs contain 107 video frames and 32 kHz stereo
audio. Timings cover generation plus MP4 mux/write, excluding model startup
and verification. Manifests and logs are in
`~/chenyb/validation/h3-a1/generation/non-turbo-t2va-tp2-dlo-50step-1344x768/`
and `~/chenyb/validation/h3-a1/generation/turbo-t2va-tp2-dlo-8step-1344x768/`,
with matching `*-tp2-dlo-*.log` files. Both DLO runs emitted the framework's
30 s Orchestrator shutdown timeout and a shared-memory resource-tracker warning
after producing their complete manifests; the same shutdown timeout appears
in the earlier TP2+DLO smoke. The worker processes exited and released their
GPUs, but clean DLO shutdown remains unverified.

All three beta5 Turbo tasks also pass at 1344×768 on TP4. T2VA produced one
107-frame MP4 with 32 kHz stereo audio in 58.12 s including mux/write. FL2VA
and Ref2VA each had one warmup and three measured requests: 64.652–64.719 s
and 59.663–59.732 s, respectively. Per-run manifests and logs are in
`~/chenyb/validation/h3-a1/generation/turbo-{t2va,fl2va,ref2va}-tp4-8step-1344x768/`
and the matching `turbo-*-tp4-8step-1344x768.log` files. Non-Turbo TP4,
FL2VA/Ref2VA under DLO and the broader profile matrix remain open; evidence and
untested combinations are tracked in
`~/chenyb/validation/h3-a1/PROFILE_MATRIX.md`.

### 5. Fixed-seed quality and latency evaluation

- [x] Define a reproducible prompt/input suite, seeds, shapes, sampling settings and metric implementations.
- [ ] Run non-Turbo against official H3 and Turbo against official H3 plus LightX2V Turbo.
- [ ] Report LPIPS, audio similarity and VBench, with metric versions and preprocessing documented.
- [ ] Report end-to-end seconds per request with hardware, warmup policy, repetitions and timing boundaries.
- [ ] Preserve side-by-side videos and per-example results, including failures and quality regressions.
- [ ] Record model/code revisions, environment, commands and artifact locations for reproduction.

The prepared suite uses three deterministic toy scenes with seeds 42/2026/7
at 1344×768 and 24 FPS, with matching inputs across each task and A/B pair.
Metrics are all-frame AlexNet LPIPS v0.1, stereo/mono CLAP embedding cosine,
and four preselected VBench custom-input dimensions: subject consistency,
background consistency, motion smoothness and aesthetic quality. The latter
are dimension scores, not a full standard-suite VBench aggregate. Two matched
generation pairs (non-Turbo T2VA seeds 42 and 2026) are complete; plan-selected
quality metrics remain pending for all 18 pairs. Protocol and self-check
evidence are in
`~/chenyb/validation/h3-a1/METRIC_PROTOCOL.md`.

The three-seed comparison plan has 36 generation commands, 18 matched metric
pairs and 36 VBench jobs. The initial plan at
`~/chenyb/validation/h3-a1/evaluation-20261003/evaluation-plan.json` is a
historical protocol artifact. The HSDP4 plan at
`~/chenyb/validation/h3-a1/evaluation-20261004/evaluation-plan.json` is also
superseded: its official FL2VA preflight OOMed. Use the regenerated TP2+DLO
resident-layers=20 plan at
`~/chenyb/validation/h3-a1/evaluation-20261004-tp2-dlo-resident20/evaluation-plan.json`;
it points at this worktree and uses the same profile for both sides. The host
territory was confirmed on 2026-10-04, so the earlier territory gate is cleared.
The official FL2VA and Ref2VA reference DiT shards total 132.6 GB. All 26
transformer payload hashes match the pinned Hub revision. An official FL2VA
HSDP4 preflight loaded the model on four L40S GPUs, but the 448×256, two-step
request ran out of memory: each rank used 44.33 GiB of 44.40 GiB and rank 3
could not allocate another 74 MiB. The failed run is recorded at
`~/chenyb/validation/h3-a1/official-fl2va-hsdp4-2step-448x256-preflight-r1.log`.

The same three official tasks then completed one 448×256, two-step request
each with TP2+DLO on GPUs 0 and 1. T2VA, FL2VA and Ref2VA all produced 107-frame
H.264 video with 32 kHz stereo AAC; each MP4 hash matched its manifest and
passed full `ffmpeg` decode. Generate-plus-mux/write times were 9.17 s, 12.42 s
and 14.57 s, respectively; each worker reported 1.81 GiB process-scoped GPU
memory after model initialization. These are task-path preflights, not matched
quality or production-latency results. Evidence and manifests are in
`~/chenyb/validation/h3-a1/generation/official-{t2va,fl2va,ref2va}-tp2-dlo-2step-448x256-r1/`.
All three DLO runs exited with status 0 but emitted the known Orchestrator
30-second shutdown timeout and shared-memory resource-tracker warning after
writing complete outputs. The ordinary restricted shell does not expose GPU
device nodes or the host `/dev/shm`; task-scoped elevated execution can access
the host GPUs and the already staged, hash-verified official weights. The three
official task-path preflights are complete. One formal-size, matched seed-42
non-Turbo T2VA pair has also completed and passed full media decode.

The official Turbo + LightX2V adapter comparison uses the H3 PEFT loader and
sets a `lora_request` on the sampling parameters. A preflight with the generic
distilled-LoRA backend, and a PEFT preflight without the request field, produced
the same MP4 SHA as unadapted official H3. With PEFT plus the explicit request,
the worker logged adapter activation and the 8-step output hash differed from
the unadapted output; the MP4 passed full decode. This 448×256 check validates
adapter application, not Turbo quality. The evaluation runner records adapter
ID, path and scale in each manifest. Evidence is in
`~/chenyb/validation/h3-a1/generation/official-turbo-lightx-t2va-tp2-dlo-active-lora-smoke-448x256/`.

The original TP2+DLO batch was interrupted after its first formal case proved
too slow with zero resident layers. A replacement plan uses DLO with 20 resident
layers and GPUs 0 and 1. It started on GPUs 0 and 1 at 12:33 UTC; it checks device occupancy before
each command and stops on its first generation or validation error. Prior
formal-size T2VA generation took about 444 seconds per request before startup.
For the first suite item, its official side completed a warmup plus three
measured seed-42 runs at 439.96, 439.70 and 437.83 s (median 439.70 s);
startup was 126.08 s and excluded. All four official MP4 hashes matched their
manifests, with 107-frame video/stereo audio shapes, and all passed full
`ffmpeg` decode. The beta5 side completed at 437.654, 437.628 and 437.354 s (median 437.628 s).
Under this matched TP2+DLO resident-layers=20 profile, the beta5 request was
0.47% faster by median elapsed time for this seed/task. This is one pair only,
not a broad performance claim. All beta5 outputs passed manifest hashes and
full `ffmpeg` decode. Both sides use seed 42, 50 steps and the same 1344×768
prompt/configuration. Evidence for both is in
`~/chenyb/validation/h3-a1/evaluation-20261004-tp2-dlo-resident20/generation/00-non-turbo-t2va-seed42/`.
Plan-selected LPIPS/CLAP/VBench on measured outputs across all 18 matched pairs
remain pending.

A second non-Turbo T2VA comparison completed at seed 2026 with the same prompt,
resolution, 50 steps and TP2+DLO resident-layers=20 profile. Official H3 warmup
was 445.936 s and measured runs were 440.493, 440.224 and 440.118 s (median
440.224 s); beta5 warmup was 441.767 s and measured runs were 439.643, 439.492
and 438.817 s (median 439.492 s). Beta5 was 0.17% lower by median request time
for this seed/task/profile. All eight MP4s matched their manifest hashes and
passed expected frame/audio shape and full-decode checks. Matched LPIPS/CLAP/
VBench results remain pending; this small timing difference is not a general
performance claim. Evidence is under
`~/chenyb/validation/h3-a1/evaluation-20261004-tp2-dlo-resident20/generation/01-non-turbo-t2va-seed2026/`.

The seed-2026 Turbo T2VA pair completed with the pinned LightX2V 8-step
adapter on official H3. Official warmup was 88.845 s and measured runs were
85.417, 85.296 and 85.028 s (median 85.296 s); beta5 warmup was 83.627 s and
runs were 80.508, 80.596 and 80.648 s (median 80.596 s). Beta5 was 5.51% lower
by median request time for this seed/task/profile. All eight MP4s passed
manifest hash, expected frame/audio shape and full-decode checks. Matched
LPIPS/CLAP/VBench remains pending; this timing is not a general performance
claim. Evidence is under
`~/chenyb/validation/h3-a1/evaluation-20261004-tp2-dlo-resident20/generation/01-turbo-t2va-seed2026/`.

The seed-2026 non-Turbo FL2VA pair completed with the suite's first/last-frame
conditioning. Official H3 warmup was 522.949 s and measured runs were 518.682,
518.647 and 518.660 s (median 518.660 s); beta5 warmup was 522.169 s and runs
were 518.547, 518.057 and 518.096 s (median 518.096 s). Beta5 was 0.11% lower
by median request time for this task/profile. The Turbo FL2VA pair, using the
pinned LightX2V 8-step adapter on official H3, measured 98.825, 98.415 and
98.414 s (median 98.415 s) after a 104.907 s warmup; beta5 measured 94.861,
94.669 and 94.602 s (median 94.669 s) after a 97.749 s warmup, 3.81% lower by
median. All 16 MP4s passed manifest-hash, 107-frame video, 32 kHz stereo-audio
shape and complete `ffmpeg` decode checks. These are narrow latency results,
not overall quality or performance claims; matched LPIPS/CLAP/VBench results
remain pending. Evidence is under
`~/chenyb/validation/h3-a1/evaluation-20261004-tp2-dlo-resident20/generation/01-{non-turbo,turbo}-fl2va-seed2026/`.

The non-Turbo Ref2VA seed-2026 pair also completed at 1344×768, 50 steps and
TP2+DLO resident-layers=20, using the same prompt and reference image. Official
H3 warmup was 485.203 s and measured runs were 480.624, 479.956 and 479.947 s
(median 479.956 s); beta5 warmup was 479.474 s and measured runs were 478.171,
476.332 and 477.584 s (median 477.584 s), 0.49% lower by median request time.
All eight MP4s passed manifest hash, expected frame/audio shape and complete
`ffmpeg` decode checks. This is a narrow latency result; matched LPIPS/CLAP/
VBench remains pending. Evidence is under
`~/chenyb/validation/h3-a1/evaluation-20261004-tp2-dlo-resident20/generation/01-non-turbo-ref2va-seed2026/`.

The Turbo Ref2VA seed-2026 pair used the pinned LightX2V 8-step adapter on
official H3 and the beta5 Turbo checkpoint as stored, at 1344×768 with
TP2+DLO resident-layers=20. Official warmup was 96.765 s and measured runs
were 92.239, 91.862 and 91.773 s (median 91.862 s); beta5 warmup was 90.168 s
and measured runs were 87.531, 87.534 and 87.699 s (median 87.534 s), 4.71%
lower by median request time. All eight MP4s passed manifest hash, expected
frame/audio shape and complete `ffmpeg` decode checks. Matched LPIPS/CLAP/VBench
remains pending; this is a narrow latency result. Evidence is under
`~/chenyb/validation/h3-a1/evaluation-20261004-tp2-dlo-resident20/generation/01-turbo-ref2va-seed2026/`.

The non-Turbo T2VA seed-7 pair completed at 1344×768, 50 steps and TP2+DLO
resident-layers=20 with the same prompt. Official H3 warmup was 445.831 s and
measured runs were 439.796, 439.479 and 439.178 s (median 439.479 s); beta5
warmup was 442.881 s and measured runs were 438.348, 438.411 and 438.100 s
(median 438.348 s), 0.26% lower by median request time. All eight MP4s passed
manifest hash, expected frame/audio shape and complete `ffmpeg` decode checks.
This is a narrow timing result; matched LPIPS/CLAP/VBench remains pending.
Evidence is under
`~/chenyb/validation/h3-a1/evaluation-20261004-tp2-dlo-resident20/generation/02-non-turbo-t2va-seed7/`.

The Turbo T2VA seed-7 pair completed at 1344×768, 8 steps and TP2+DLO
resident-layers=20. Official H3 used the pinned LightX2V Turbo adapter via
PEFT; its warmup was 89.180 s and measured runs were 85.574, 85.353 and 85.245 s
(median 85.353 s). Beta5 warmup was 82.959 s and measured runs were 80.233,
80.151 and 80.100 s (median 80.151 s), 6.09% lower by median request time.
All eight MP4s passed manifest hash, expected frame/audio shape, ffprobe and
complete `ffmpeg` decode checks. This is a narrow timing result; matched
LPIPS/CLAP/VBench remains pending. Evidence is under
`~/chenyb/validation/h3-a1/evaluation-20261004-tp2-dlo-resident20/generation/02-turbo-t2va-seed7/`.

For the first non-Turbo FL2VA seed-42 case, official H3 and beta5 completed with
the same TP2+DLO resident-layers=20 profile, suite prompt and first/last images.
Official warmup was 526.305 s and measured runs were 519.705, 519.736 and
519.223 s (median 519.705 s); beta5 warmup was 521.184 s and measured runs
were 517.461, 517.929 and 518.418 s (median 517.929 s). Model startup was
113.006 s for official H3 and excluded. Beta5 was 0.34% faster by median
request time for this seed/task/profile. All eight MP4s matched manifest hashes
and passed full decode with 107 frames and stereo audio. This is one matched
latency result, not a broad performance or quality claim. Evidence is under
`~/chenyb/validation/h3-a1/evaluation-20261004-tp2-dlo-resident20/generation/00-non-turbo-fl2va-seed42/`.

The official Turbo T2VA side for seed 42 also completed with the pinned
LightX2V 8-step adapter loaded and activated through H3's PEFT path. Its warmup
was 89.126 s and three measured runs were 84.773, 84.884 and 84.616 s (median
84.773 s); startup was 114.36 s and excluded. The four MP4s matched manifest
hashes and passed full decode. The beta5 side completed with 84.463 s warmup
and measured runs of 80.631, 80.519 and 80.681 s (median 80.631 s). Its median
was 4.89% faster in this single matched seed/task/profile. Both sides' eight
MP4s passed manifest hash and full decode checks. This is a narrow timing result,
not an overall quality claim; plan-selected LPIPS/CLAP/VBench scores are still
pending. Evidence is under
`~/chenyb/validation/h3-a1/evaluation-20261004-tp2-dlo-resident20/generation/00-turbo-t2va-seed42/`.

The matched Turbo FL2VA seed-42 case has also completed with the pinned
LightX2V 8-step adapter. The official adapter loaded through H3's PEFT path and
worker logs confirm it was activated for task `fl2v`. Official warmup was
104.800 s and measured runs were 98.949, 98.764 and 98.362 s (median 98.764 s);
beta5 warmup was 98.369 s and measured runs were 95.170, 94.854 and 94.781 s
(median 94.854 s). Startup was 118.738 s for official H3 and excluded. Beta5
was 3.96% faster by median request time for this seed/task/profile. All eight
MP4s passed manifest hash, expected shape/audio and full-decode checks. This is
a narrow latency result, not an overall quality or performance claim; planned
LPIPS/CLAP/VBench for this pair remain pending. Evidence is in
`~/chenyb/validation/h3-a1/evaluation-20261004-tp2-dlo-resident20/generation/00-turbo-fl2va-seed42/`.

For the first non-Turbo Ref2VA seed-42 case, official H3 completed with the
suite prompt and matching appearance image at 1344×768, 50 steps and TP2+DLO
resident-layers=20. Warmup was 486.670 s and measured runs were 481.751,
481.802 and 481.753 s (median 481.753 s); startup was 124.435 s and excluded.
All four MP4s matched manifest hashes and passed full decode with 107 frames
and stereo audio. The beta5 side completed with a 482.945 s warmup and measured
runs of 480.237, 480.575 and 480.886 s (median 480.575 s); startup is excluded.
This is 0.24% lower median request time than official H3 for this seed/task/
profile. All eight MP4s matched manifest hashes and passed full decode with 107
frames and stereo audio. Matched LPIPS/CLAP/VBench results are pending. Evidence
is under
`~/chenyb/validation/h3-a1/evaluation-20261004-tp2-dlo-resident20/generation/00-non-turbo-ref2va-seed42/`.

For Turbo Ref2VA at the same resolution and profile, official H3 used the
pinned LightX2V 8-step adapter via PEFT, recorded in the request manifest.
Official warmup was 95.641 s and measured runs were 92.388, 92.371 and 92.309 s
(median 92.371 s). Beta5 warmup was 91.807 s and runs were 88.373, 88.013 and
87.881 s (median 88.013 s), 4.72% lower by median request time for this single
seed/task/profile. All eight MP4s passed manifest hash, expected frame/audio
shape and full-decode checks. Matched LPIPS/CLAP/VBench results are pending.
Evidence is under
`~/chenyb/validation/h3-a1/evaluation-20261004-tp2-dlo-resident20/generation/00-turbo-ref2va-seed42/`.

An exploratory VBench pass now covers one generated video for each beta5 variant
and task. The table reports single-video scores for subject consistency,
background consistency, motion smoothness and aesthetic quality, in that order;
these are not matched-seed comparisons and do not establish that either model is
better. Turbo FL2VA's lower subject-consistency score needs paired evaluation;
it does not by itself establish a motion regression. One paired comparison is
now complete. For non-Turbo T2VA, seed 42, 50 steps, 1344×768 and TP2+DLO
resident-layers=20, mean all-frame LPIPS is 0.2688; CLAP
left/right/mono cosine is 0.1939/0.1715/0.1592. Official vs beta5 VBench scores
(subject, background, motion, aesthetic) are 0.9794/0.9708/0.9965/0.4143 and
0.9545/0.8900/0.9964/0.4350. This single toy scene measures output differences,
not overall quality. Full metrics, per-frame LPIPS and provenance are under
`~/chenyb/validation/h3-a1/metrics/matched/seed42-t2va/`; aligned video is
`side-by-side.mp4` there. This is a separate one-off quality pair; the plan-selected
measured-run quality scores and the other 17 pairs remain pending. Raw per-video
results and provenance are in `~/chenyb/validation/h3-a1/metrics/`.

| Variant / task | Subject | Background | Motion | Aesthetic |
| --- | ---: | ---: | ---: | ---: |
| non-Turbo T2VA | 0.9557 | 0.8863 | 0.9965 | 0.4358 |
| non-Turbo FL2VA | 0.9798 | 0.9314 | 0.9957 | 0.3979 |
| non-Turbo Ref2VA | 0.9958 | 0.9703 | 0.9967 | 0.5106 |
| Turbo T2VA | 0.9117 | 0.9170 | 0.9945 | 0.4121 |
| Turbo FL2VA | 0.7630 | 0.8985 | 0.9955 | 0.4983 |
| Turbo Ref2VA | 0.9823 | 0.9560 | 0.9966 | 0.4818 |

LPIPS measures output differences and does not by itself establish better quality.
No numerical quality threshold is specified in the acceptance criteria; report
measured results and limitations without inventing a pass threshold.

### 6. Final documentation and handoff

- [ ] Write verified installation and serving instructions for both checkpoint variants and all three tasks.
- [ ] Document base-component sources, supported layouts, Turbo settings and validated deployment profiles.
- [ ] Publish a reviewable validation report with reproduction commands and evidence locations.
- [x] Review checkpoint/base-component license requirements and document the source declarations and material conditions below.
- [x] Review the scoped diff and prepare a handoff with environment, test results and remaining limitations.
- [ ] After all A1 acceptance work is complete, replace this temporary TODO with final usage documentation
  and a serving-recipe link; remove machine-specific development notes from the README.

#### Third-party checkpoint licenses

The beta5 files are third-party artifacts and are not included with vLLM-Omni.
The [TenStrip model card](https://huggingface.co/TenStrip/10Eros-Max) says the
MiniMax H3 Community License applies and that transferred Krea 2, LTX 2.3 and
Wan 2.2 material remains subject to its source-model community license. That is
the checkpoint author's declaration; it does not determine the legal treatment
of every merged tensor.

Review the [MiniMax H3 license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
before use: its defined territory excludes the United States, European Union,
United Kingdom and Republic of Korea; it restricts use of H3 outputs to improve
other AI models; commercial products/services above USD 20M annual revenue need
prior written authorization; and commercial services using H3 must prominently
display the MiniMax H3 name. Hosted services and distributions also have
downstream-user and notice obligations.

The [Krea 2 Community License](https://github.com/krea-ai/krea-2/blob/main/docs/KREA-2-COMMUNITY-LICENSE)
sets a USD 1M company-wide trailing-twelve-month revenue threshold for
commercial use under its community terms and specifies content-filtering and
derivative distribution requirements. The [LTX 2.x license](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x)
requires entities with annual revenue of at least USD 10M to obtain a paid
commercial-use agreement for commercial use of LTX-2.x and derivatives, and
sets additional use and distribution conditions. The [Wan 2.2 model card](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)
declares Apache-2.0. Review each complete license for the intended use; these
notes are not legal advice. The vLLM-Omni Apache-2.0 software license does not
change the checkpoint or base-weight license terms.

<!-- END H3 A1 BRANCH TODO -->

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/logos/vllm-omni-logo.png">
    <img alt="vllm-omni" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/logos/vllm-omni-logo.png" width=55%>
  </picture>
</p>
<h3 align="center">
Easy, fast, and cheap omni-modality model serving for everyone
</h3>

<p align="center">
| <a href="https://vllm-omni.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href="https://deepwiki.com/vllm-project/vllm-omni"><b>DeepWiki</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> | <a href="docs/assets/WeChat.jpg"><b>WeChat</b></a> | <a href="https://arxiv.org/abs/2602.02204"><b>Paper</b></a> | <a href="https://docs.google.com/presentation/d/1aPj0OGl_-ZVoib-Qne5dGDAlrRFB-PdHl6E-EE99g8E/edit?usp=sharing"><b>Slides</b></a> |
</p>

---

*Latest News* 🔥

- [2026/09] We released [0.30.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.30.0), rebased onto vLLM 0.30.0, featuring a unified full-duplex serving framework around engine-owned sessions for [MiniCPM-o 4.5](recipes/OpenBMB/MiniCPM-o-4_5.md) and AURA, native cross-stage KV and multimodal payload transfer (Mooncake AR-to-DiT handoff and NIXL connectors), interactive world-model serving with [LingBot World](recipes/Robbyant/LingBot-World-2.0.md), and realtime [MiniMax H3](recipes/MiniMaxAI/MiniMax-H3.md) Turbo inference on NVIDIA Blackwell and Ascend 950.
- [2026/08] We released [0.28.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.28.0), featuring production-ready [MiniMax H3](recipes/MiniMaxAI/MiniMax-H3.md) serving on GPU and NPU, a unified AR/DiT paged KV cache runtime, and enhanced realtime full-duplex serving for the [MiniCPM-o series](recipes/OpenBMB/MiniCPM-o-4_5.md).
- [2026/08] [VeRL-Omni](https://github.com/verl-project/verl-omni) `v0.2.0` is released: faster diffusion RL powered by vLLM-Omni (request-level/step-wise batching with FA3), rebuilt Qwen3-Omni multimodal training (DPO & GSPO), plus LTX-2.3, Qwen-Image-Edit support and more. See the [release notes](https://github.com/verl-project/verl-omni/releases/tag/v0.2.0).
- [2026/08] We released [0.26.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.26.0) - aligned with the vLLM 0.26 release line, featuring [MiniMax H3](recipes/MiniMaxAI/MiniMax-H3.md) joint video/audio generation, an experimental full-duplex realtime runtime for [MiniCPM-o 4.5](recipes/OpenBMB/MiniCPM-o-4_5.md), distributed layerwise diffusion offload, and broader model, hardware, streaming, TTS, and quantization support.
- [2026/07] We released [0.24.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.24.0) - aligned with the vLLM 0.24 release line, expanding production-ready coverage across TTS, speech, diffusion, image/video generation, and robot-policy serving, with major Omni stage runtime refactoring, diffusion request-level batching, async output materialization, quantization/cache/memory improvements, and broad CUDA/ROCm/XPU/NPU support.
- [2026/06] Starting with [0.14.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.14.0), vLLM-Omni publishes a stable release aligned with every even-numbered upstream vLLM minor version. [0.16.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.16.0), [0.18.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.18.0), [0.20.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.20.0), and [0.22.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.22.0) continued this cadence, expanding omni and world-model support with [NVIDIA Cosmos3](recipes/cosmos3/Cosmos3-Nano.md) and DreamZero, adding models such as MiniCPM-o 4.5, MOSS-TTS, and Lance, and advancing TTS, diffusion, distributed execution, quantization, RL integration through [VeRL-Omni](https://github.com/verl-project/verl-omni), and CUDA/ROCm/MUSA/NPU/XPU coverage.
- [2026/03] Check out our first public [project deepdive](https://youtu.be/sgwNfsNnR9I) at the vLLM Hong Kong Meetup!
- [2025/11] vLLM community officially released [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) in order to support omni-modality models serving.

---

## About

[vLLM](https://github.com/vllm-project/vllm) was originally designed to support large language models for text-based autoregressive generation tasks. vLLM-Omni is a framework that extends its support for omni-modality model inference and serving:

- **Omni-modality**: Text, image, audio, video, and action data processing
- **Non-autoregressive Architectures**: extend the AR support of vLLM to Diffusion Transformers (DiT) and other parallel generation models
- **Heterogeneous outputs**: from traditional text generation to multimodal and action outputs

<p align="center">
  <picture>
    <img alt="vllm-omni" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/omni-modality-model-architecture.png" width=55%>
  </picture>
</p>

vLLM-Omni is fast with:

- State-of-the-art AR support by leveraging efficient KV cache management from vLLM
- Pipelined stage execution overlapping for high throughput performance
- Fully disaggregation based on OmniConnector and dynamic resource allocation across stages

vLLM-Omni is flexible and easy to use with:

- Heterogeneous pipeline abstraction to manage complex model workflows
- Seamless integration with popular Hugging Face models
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- [Full-duplex realtime serving](docs/serving/realtime_duplex_api.md) with streaming audio input and output

vLLM-Omni seamlessly supports most popular open-source models on HuggingFace, including:

- **Omni-modality models** (e.g. Qwen3-Omni, MiniCPM-o 4.5, Cosmos3, HunyuanImage, BAGEL)
- **TTS models** (e.g. Qwen3-TTS, Tencent AuK, Breeze-TTS-2, CosyVoice3)
- **Diffusion models** — image, video, and audio generation (e.g. MiniMax H3, LingBot World, MAGI-2, LTX-2.5, Wan2.2)
- **Robot-policy and action models** (e.g. π0.5, GR00T-N1.7, DreamZero-DROID, InternVLA-A1)

## Getting Started

Visit our [documentation](https://vllm-omni.readthedocs.io/en/latest/) to learn more.

- [Installation](https://vllm-omni.readthedocs.io/en/latest/getting_started/installation/)
- [Quickstart](https://vllm-omni.readthedocs.io/en/latest/getting_started/quickstart/)
- [Nightly CUDA Docker images](https://hub.docker.com/r/vllm/vllm-omni/tags?name=nightly) built from the latest `main`
- [List of Supported Models](https://vllm-omni.readthedocs.io/en/latest/models/supported_models/)
- [Deployment Recipes](https://recipes.vllm.ai) for vLLM-Omni model serving

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM-Omni](https://vllm-omni.readthedocs.io/en/latest/contributing/) for how to get involved.

## Citation

If you use vLLM-Omni for your research, please cite our [paper](https://arxiv.org/abs/2602.02204):

```bibtex
@article{yin2026vllmomni,
  title={vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models},
  author={Peiqi Yin, Jiangyun Zhu, Han Gao, Chenguang Zheng, Yongxiang Huang, Taichang Zhou, Ruirui Yang, Weizhi Liu, Weiqing Chen, Canlin Guo, Didan Deng, Zifeng Mo, Cong Wang, James Cheng, Roger Wang, Hongsheng Liu},
  journal={arXiv preprint arXiv:2602.02204},
  year={2026}
}
```

## Join the Community

Feel free to ask questions, provide feedbacks and discuss with fellow users of vLLM-Omni in `#sig-omni` slack channel at [slack.vllm.ai](https://slack.vllm.ai) or vLLM user forum at [discuss.vllm.ai](https://discuss.vllm.ai).

## Star History

<a href="https://www.star-history.com/?repos=vllm-project%2Fvllm-omni&type=date&legend=top-left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=vllm-project/vllm-omni&type=date&theme=dark&legend=top-left&sealed_token=ExgLDZJoQEg27Zfhhut2LqN0GYO6Fw2PWLwPE6JYBUp2BgM3hmsYlwaIVopnUEfbRXidQ4nisumrTdKYydiKhy1SZXipw47qY2_tiUDhCpsPXeXtPuEVKVzBwKs3pw0tiHsJgtSfwXx5yjHXck0Y2SblzFWeJYCkTe1WLGTbUAOIETjXXQJjyCGZvKz5" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=vllm-project/vllm-omni&type=date&legend=top-left&sealed_token=ExgLDZJoQEg27Zfhhut2LqN0GYO6Fw2PWLwPE6JYBUp2BgM3hmsYlwaIVopnUEfbRXidQ4nisumrTdKYydiKhy1SZXipw47qY2_tiUDhCpsPXeXtPuEVKVzBwKs3pw0tiHsJgtSfwXx5yjHXck0Y2SblzFWeJYCkTe1WLGTbUAOIETjXXQJjyCGZvKz5" />
    <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=vllm-project/vllm-omni&type=date&legend=top-left&sealed_token=ExgLDZJoQEg27Zfhhut2LqN0GYO6Fw2PWLwPE6JYBUp2BgM3hmsYlwaIVopnUEfbRXidQ4nisumrTdKYydiKhy1SZXipw47qY2_tiUDhCpsPXeXtPuEVKVzBwKs3pw0tiHsJgtSfwXx5yjHXck0Y2SblzFWeJYCkTe1WLGTbUAOIETjXXQJjyCGZvKz5" />
  </picture>
</a>

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file.
