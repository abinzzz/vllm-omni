# MiniMax-H3 beta5 BF16 single-file checkpoints

vLLM-Omni can load the TenStrip/10Eros-Max beta5 BF16 Turbo and non-Turbo
checkpoints directly from a local safetensors file. The selected file supplies
the H3 DiT; the remaining components are loaded from the pinned MiniMax H3
base repository. Both variants support T2VA, FL2VA, and Ref2VA. The loader
executes the checkpoints' compressed AdaLN representation as stored.

This support covers beta5 BF16. INT8 and W4A8 checkpoints are not supported by
this change. Review the [TenStrip checkpoint card](https://huggingface.co/TenStrip/10Eros-Max)
and [MiniMax H3 license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
including applicable source-model terms, before use.

## Install and obtain the checkpoint

Install vLLM-Omni from this checkout using the project's standard environment
setup. Then authenticate to Hugging Face if required and download one beta5
file:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
hf auth login

export CHECKPOINT_DIR="$HOME/models/10eros-max"
mkdir -p "$CHECKPOINT_DIR"
hf download TenStrip/10Eros-Max 10Eros_Max_h3_hybrid_beta5.safetensors \
  --revision 8a198588c8870ab0d613b3492a3150d091c8c2dd \
  --local-dir "$CHECKPOINT_DIR"
```

For Turbo, download `10Eros_Max_h3_TURBO-hybrid_beta5.safetensors` instead.
The beta5 source revision is pinned above. The base components are resolved
from `MiniMaxAI/MiniMax-H3` revision
`42ed227ee7df40d41602854ae760620d6eb651fe`; Hugging Face access must be
available to the serving process. Override the base only with a compatible H3
component repository and revision.

## Start a server

The following four-GPU HSDP configuration matches the A1 functional profile.
Use a task-specific server: `--task-type fl2va` serves T2VA and FL2VA;
`--task-type ref2va` serves Ref2VA. Both servers use the same checkpoint file,
and requests select the operation through `extra_params.task`. A combined
`--task-type auto` server was not validated.

```bash
export CHECKPOINT="$CHECKPOINT_DIR/10Eros_Max_h3_hybrid_beta5.safetensors"
export PORT=8091

PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve "$CHECKPOINT" \
  --omni \
  --model-class-name MiniMaxH3Pipeline \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --task-type fl2va \
  --custom-pipeline-args '{"base_model":"MiniMaxAI/MiniMax-H3","base_revision":"42ed227ee7df40d41602854ae760620d6eb651fe"}' \
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

This task-specific HSDP4 profile passed HTTP requests for T2VA and FL2VA on the
validation host. Start a second instance with `--task-type ref2va` and a distinct
port for Ref2VA. The combined `--task-type auto` startup was not validated. At
request time, use Euler with guidance scale `1`, video flow shift `12`, and audio
flow shift `3`. The non-Turbo checkpoint was validated at 50 steps; the merged
Turbo checkpoint was validated at 8 Euler steps. Turbo deltas are
already merged into that file: do not load the LightX2V Turbo adapter on top.

The [MiniMax H3 serving recipe](recipes/MiniMaxAI/MiniMax-H3.md) documents the
HTTP request fields and examples for [T2VA](recipes/MiniMaxAI/MiniMax-H3.md#1-t2va-text-to-video-and-audio),
[FL2VA](recipes/MiniMaxAI/MiniMax-H3.md#2-fl2va-first-frame-to-video-and-audio),
and [Ref2VA](recipes/MiniMaxAI/MiniMax-H3.md#3-ref2va-image-only-imageaudio-or-mixed-references).
Set `extra_params.task` to `t2va`, `fl2va`, or `ref2va`; pass FL2VA keyframes
or Ref2VA references with the recipe's `input_reference` fields.

## Validated scope and limits

Both checkpoints produced valid 1344×768 video-plus-audio outputs for all
three tasks. The fixed-seed comparison covers three seeds, 18 task/variant
pairs, LPIPS, stereo-channel and mono CLAP similarity, four VBench dimensions,
and repeated request timing against official H3 (using the pinned LightX2V
adapter for Turbo). The detailed results, validation commands, environment,
and evidence locations are in the [A1 validation report](docs/source/features/diffusion/minimax_h3_beta5_validation.md).

Validated beta5 deployment profiles are HSDP4 for all six variant/task cases,
TP2 with distributed layerwise offload (DLO) and 20 resident layers for the
matched A/B suite, and TP2+DLO with zero resident layers for one full-resolution
request per variant/task. Turbo also passed TP4 for all three tasks. Non-Turbo
TP4, pre-sharded HSDP, other GPU counts and offload combinations, and concurrent
independent DLO requests have not been established. DLO runs emitted an
Orchestrator shutdown timeout and shared-memory tracker warning after successful
inference; see the profile matrix in the validation report.

## H3 serving recipe

For the general H3 model, memory profiles, API fields, and request examples,
see the [MiniMax H3 recipe](recipes/MiniMaxAI/MiniMax-H3.md).


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
