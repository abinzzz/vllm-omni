<!-- BEGIN H3 A1 BRANCH TODO -->

# MiniMax-H3 A1: BF16 single-file checkpoint support

This branch tracks implementation and validation of TenStrip/10Eros-Max beta5
BF16 Turbo and non-Turbo checkpoints in vLLM-Omni. Support is **in development;
end-to-end serving is not validated yet**. Keep this checklist current as work progresses; mark an item
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
- [ ] Establish a compatible complete inference environment, including compiled vLLM extensions.
- [x] Inventory checkpoint storage requirements and available GPUs before downloads or inference.

The CPU baseline only checks existing behavior; it is not evidence of A1 support.
Machine-specific logs and environment notes live outside the repository in
`~/chenyb/validation/h3-a1/`, including `HANDOFF.md` and `baseline-modulation.log`.
GPU occupancy must be checked again before each run.

### 1. Checkpoint and reference investigation

- [x] Pin beta5 BF16 Turbo/non-Turbo filenames, revisions and checksums, and base H3 component revisions.
- [x] Inspect checkpoint metadata, tensor names, dtypes and shapes before large downloads where possible.
- [x] Identify which components are stored in the file and which must come from base H3.
- [x] Read the reference implementation and document compressed AdaLN computation and parameter mapping.
- [ ] Verify how each checkpoint supports T2VA, FL2VA and Ref2VA, including partition-specific differences.
- [ ] Confirm Turbo step counts, sigma schedules, guidance settings and already-merged adapters.
- [x] Finalize the loading interface and compatibility design from this evidence.

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
- [ ] Preserve the official H3 loading path and integrate with AdaLN caching, offload and parallel loading.
- [ ] Apply validated Turbo sampling settings without applying merged adapters a second time.

Pre-sharded HSDP loading currently rejects H3 checkpoints because the shared loader
requires runtime-layout weights and cannot apply H3 tensor transforms. Ordinary
HSDP loading is tested above; production checkpoint profiles still require validation.

### 3. Focused regression tests

- [x] Test weight mapping and base-component selection with small synthetic checkpoints.
- [x] Test malformed, incomplete and unsupported checkpoints and clear error reporting.
- [x] Compare compressed AdaLN outputs with the reference computation on small tensors, documenting tolerances.
- [ ] Cover task/partition selection and Turbo schedule handling.
- [ ] Run applicable official H3, single-file loader and configuration regression tests.
- [ ] Run repository formatting and lint checks for changed files.

### 4. End-to-end generation

Start with small-shape smoke tests, then repeat at the recorded acceptance shapes.
Complete each row only after preserving the command, configuration, seed, logs
and output evidence.

| Variant | T2VA | FL2VA | Ref2VA |
| --- | --- | --- | --- |
| beta5 BF16 non-Turbo | Smoke passed; acceptance pending | Pending | Pending |
| beta5 BF16 Turbo | Pending | Pending | Pending |

The non-Turbo T2VA smoke produced a decoded 448×256 MP4 with 107 frames
and 32 kHz stereo audio at two Euler evaluations on two L40S GPUs with TP2
and rank-local DLO. Request time including mux/write was 11.69 s, excluding
129.88 s startup. This is a smoke result, not an acceptance-shape quality claim.
Engine shutdown logged cleanup timeouts, but host checks confirmed workers
exited and GPU memory was released; lifecycle validation remains pending.

- [ ] Complete all six generation cases above.
- [ ] Verify MP4 decoding, dimensions, frame count, duration and audio track for every case.
- [ ] Verify that first/last-frame and reference conditioning enter the intended inference path.
- [ ] Verify actual checkpoint weight consumption for each task.
- [ ] Validate the parallelism and offload profile matrix and record unsupported or untested combinations.

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
are dimension scores, not a full standard-suite VBench aggregate. Actual
checkpoint results remain pending. Protocol and self-check evidence are in
`~/chenyb/validation/h3-a1/METRIC_PROTOCOL.md`.

LPIPS measures output differences and does not by itself establish better quality.
No numerical quality threshold is specified in the acceptance criteria; report
measured results and limitations without inventing a pass threshold.

### 6. Final documentation and handoff

- [ ] Write verified installation and serving instructions for both checkpoint variants and all three tasks.
- [ ] Document base-component sources, supported layouts, Turbo settings and validated deployment profiles.
- [ ] Publish a reviewable validation report with reproduction commands and evidence locations.
- [ ] Review checkpoint/base-component license requirements and document applicable usage conditions.
- [ ] Review the scoped diff and prepare a handoff with environment, test results and remaining limitations.
  The contributor will open the PR and post comments.
- [ ] After all A1 acceptance work is complete, replace this temporary TODO with final usage documentation
  and a serving-recipe link; remove machine-specific development notes from the README.

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
