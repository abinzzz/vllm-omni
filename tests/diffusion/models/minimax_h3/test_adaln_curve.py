# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def tp1(monkeypatch):
    from vllm.distributed import parallel_state

    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0))


def test_curve_interpolates_and_clamps_each_stream_timestep():
    from vllm_omni.diffusion.models.minimax_h3.adaln_curve import MiniMaxH3AdalnCurve

    curve = MiniMaxH3AdalnCurve(3, 2)
    curve.table.data.copy_(torch.tensor([[0.0, 4.0], [2.0, 8.0], [10.0, -4.0]]))
    times = torch.tensor([-0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1], dtype=torch.float64)
    expected = torch.tensor([[0, 4], [0, 4], [1, 6], [2, 8], [6, 2], [10, -4], [10, -4]], dtype=torch.float32)
    torch.testing.assert_close(curve(times), expected, rtol=0, atol=0)


@pytest.mark.parametrize("grid, width", [(0, 2), (1, 2), (3, 0), (3, -1)])
def test_curve_rejects_invalid_table_dimensions(grid, width):
    from vllm_omni.diffusion.models.minimax_h3.adaln_curve import MiniMaxH3AdalnCurve

    with pytest.raises(ValueError, match="curve"):
        MiniMaxH3AdalnCurve(grid, width)


@pytest.mark.parametrize("expand, modalities", [(6, 3), (2, 1)])
def test_curve_projection_preserves_fp32_and_does_not_apply_silu(expand, modalities):
    arch = h3.MiniMaxH3DiTArchConfig(hidden_size=2, time_embed_dim=2, adaln_curve_grid=3)
    proj = h3.MiniMaxH3AdalnProj(
        arch, expand * modalities * 2, None, expand_ratio=expand, modality_num=modalities, prefix="test"
    )
    assert proj.linear.weight.dtype == torch.float32
    proj.linear.weight.data.fill_(1)
    proj.linear.bias.data.fill_(0.125)
    coords = torch.tensor([[-2.0, 1.001], [0.001, 2.0]], dtype=torch.float32)
    actual = proj(coords)
    expected = torch.tensor([-0.874, 2.126]).repeat_interleave(modalities).unsqueeze(1).expand(-1, 2)
    for part in actual:
        assert part.dtype == torch.float32
        torch.testing.assert_close(part, expected, rtol=0, atol=3e-7)


def test_dense_projection_keeps_original_silu_and_bf16():
    arch = h3.MiniMaxH3DiTArchConfig(hidden_size=2, time_embed_dim=2)
    proj = h3.MiniMaxH3AdalnProj(arch, 4, None, expand_ratio=2, modality_num=1, prefix="test")
    proj.linear.weight.data.fill_(1)
    proj.linear.bias.data.zero_()
    coords = torch.tensor([[-2.0, 1.001]])
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(coords).to(torch.bfloat16), proj.linear.weight, proj.linear.bias
    )
    assert proj.linear.weight.dtype == torch.bfloat16
    torch.testing.assert_close(torch.cat(proj(coords), dim=-1), expected, rtol=0, atol=0)
