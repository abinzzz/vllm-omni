# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from safetensors.torch import save_file

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def checkpoint_tensors():
    # One DiT block and one text-refiner block; independent hand-sized fixture.
    shapes = {
        "adaln_t_table": (3, 2),
        "adaln_basis": (2, 4),
        "adaln_mean": (4,),
        "video_patch_proj.weight": (8, 8),
        "video_patch_proj.bias": (8,),
        "audio_patch_proj.weight": (8, 2),
        "audio_patch_proj.bias": (8,),
        "condition_proj.weight": (8, 6),
        "condition_proj.bias": (8,),
        "rope.inv_freq": (2,),
        "token_refiner.final_norm.weight": (8,),
        "blocks.0.adaln_proj.linear.weight": (144, 2),
        "blocks.0.adaln_proj.linear.bias": (144,),
        "final_layer.adaln_proj.linear.weight": (16, 2),
        "final_layer.adaln_proj.linear.bias": (16,),
        "final_layer.norm.weight": (8,),
        "final_layer.video_out.weight": (8, 8),
        "final_layer.video_out.bias": (8,),
        "final_layer.audio_out.weight": (2, 8),
        "final_layer.audio_out.bias": (2,),
    }
    for prefix in ("blocks.0", "token_refiner.blocks.0"):
        for suffix, shape in {
            "norm1.weight": (8,),
            "norm2.weight": (8,),
            "attn.q_norm.weight": (4,),
            "attn.k_norm.weight": (4,),
            "attn.qkv_proj.weight": (24, 8),
            "attn.out_proj.weight": (8, 8),
            "mlp.fc1.weight": (32, 8),
            "mlp.fc2.weight": (8, 16),
        }.items():
            shapes[f"{prefix}.{suffix}"] = shape
    return {name: torch.zeros(shape, dtype=torch.bfloat16) for name, shape in shapes.items()}


def test_inspect_compressed_checkpoint_derives_native_architecture(tmp_path, checkpoint_tensors):
    from vllm_omni.diffusion.models.minimax_h3.single_file import MiniMaxH3SingleFileSpec

    path = tmp_path / "hybrid.safetensors"
    save_file(checkpoint_tensors, path)
    spec = MiniMaxH3SingleFileSpec.from_file(path)
    config = spec.transformer_config
    assert config["hidden_size"] == 8
    assert config["num_layers"] == 1
    assert config["token_refiner_num_layers"] == 1
    assert config["num_attention_heads"] == 2
    assert config["attention_head_dim"] == 4
    assert config["time_embed_dim"] == 2
    assert config["adaln_curve_grid"] == 3
    assert config["latents_dim"] == 2
    assert config["audio_latents_dim"] == 2
    assert config["text_dim"] == 6
    assert config["ffn_hidden_size"] == 16


@pytest.mark.parametrize(
    "change, message",
    [
        ("missing_projection", "missing"),
        ("bad_projection_shape", "shape"),
        ("quantized", "quantized"),
        ("unknown_tensor", "unsupported"),
        ("missing_block", "contiguous"),
        ("invalid_grid", "curve"),
        ("dense_embedder", "unsupported"),
        ("incomplete_basis", "basis"),
    ],
)
def test_inspect_rejects_incomplete_or_unsupported_files(tmp_path, checkpoint_tensors, change, message):
    from vllm_omni.diffusion.models.minimax_h3.single_file import MiniMaxH3SingleFileSpec

    tensors = checkpoint_tensors
    if change == "missing_projection":
        del tensors["final_layer.audio_out.weight"]
    elif change == "bad_projection_shape":
        tensors["blocks.0.adaln_proj.linear.weight"] = torch.zeros(144, 4, dtype=torch.bfloat16)
    elif change == "quantized":
        tensors["blocks.0.attn.qkv_proj.weight"] = tensors["blocks.0.attn.qkv_proj.weight"].to(torch.int8)
    elif change == "unknown_tensor":
        tensors["text_encoder.unexpected.weight"] = torch.ones(1, dtype=torch.bfloat16)
    elif change == "missing_block":
        tensors.update(
            {
                key.replace("blocks.0.", "blocks.2."): value.clone()
                for key, value in list(tensors.items())
                if key.startswith("blocks.0.")
            }
        )
    elif change == "invalid_grid":
        tensors["adaln_t_table"] = torch.zeros(1, 2, dtype=torch.bfloat16)
    elif change == "dense_embedder":
        tensors["time_embedder.proj_in.weight"] = torch.ones(4, 4, dtype=torch.bfloat16)
    elif change == "incomplete_basis":
        del tensors["adaln_mean"]
    path = tmp_path / "invalid.safetensors"
    save_file(tensors, path)
    with pytest.raises(ValueError, match=message):
        MiniMaxH3SingleFileSpec.from_file(path)


def test_inspect_accepts_curve_without_optional_adapter_basis(tmp_path, checkpoint_tensors):
    from vllm_omni.diffusion.models.minimax_h3.single_file import MiniMaxH3SingleFileSpec

    del checkpoint_tensors["adaln_basis"]
    del checkpoint_tensors["adaln_mean"]
    path = tmp_path / "comfy.safetensors"
    save_file(checkpoint_tensors, path)
    assert MiniMaxH3SingleFileSpec.from_file(path).transformer_config["time_embed_dim"] == 2
