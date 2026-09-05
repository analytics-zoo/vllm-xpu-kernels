# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile/padded routing slots must never index expert tables at -1."""

import pytest
import torch

import tests.register_ops as ops


@pytest.mark.parametrize("geometry", [(16, 8, 2, 16), (8192, 256, 8, 128)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("all_invalid", [True, False])
@pytest.mark.parametrize("mapped", [False, True])
def test_moe_lora_invalid_routes(geometry, dtype, all_invalid, mapped):
    tokens, experts, topk, block = geometry
    slots = 2
    ids = torch.arange(tokens * topk, dtype=dtype).view(tokens, topk)
    ids %= experts
    if all_invalid:
        ids.fill_(-1)
    else:
        ids.view(-1)[::3] = -1
    mapping = torch.arange(tokens, dtype=torch.int32) % slots
    expert_map = torch.arange(experts - 1, -1, -1, dtype=torch.int32)
    if mapped:
        expert_map[::5] = -1
    else:
        expert_map = None
    capacity = ids.numel() + experts * (block - 1)
    blocks = (capacity + block - 1) // block
    sorted_ids = torch.full((slots * capacity, ),
                            -7,
                            dtype=torch.int32,
                            device="xpu")
    expert_ids = torch.full((slots * blocks, ),
                            -2,
                            dtype=torch.int32,
                            device="xpu")
    counts = torch.full((slots, ), -13, dtype=torch.int32, device="xpu")
    with torch.inference_mode():
        ops.moe_lora_align_block_size(
            ids.to("xpu"), mapping.to("xpu"), experts, block, slots, capacity,
            blocks, sorted_ids, expert_ids, counts,
            torch.tensor([1, 1, 0], dtype=torch.int32, device="xpu"),
            torch.tensor([0, 1, -1], dtype=torch.int32, device="xpu"),
            expert_map.to("xpu") if mapped else None)
        torch.xpu.synchronize()
    sorted_ids = sorted_ids.cpu().view(slots, capacity)
    expert_ids = expert_ids.cpu().view(slots, blocks)
    counts = counts.cpu()
    routed_ids = ids.flatten().clone()
    if mapped:
        valid = routed_ids >= 0
        routed_ids[valid] = expert_map[routed_ids[valid].long()].to(dtype)
    for slot in range(slots):
        offset = 0
        for expert in range(experts):
            expected = torch.where(
                (routed_ids == expert)
                & (mapping.repeat_interleave(topk) == slot))[0]
            padded = (expected.numel() + block - 1) // block * block
            actual = sorted_ids[slot, offset:offset + padded]
            actual = actual[actual != ids.numel()].long().sort().values
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.all(expert_ids[slot,
                                        offset // block:(offset + padded) //
                                        block] == expert)
            offset += padded
        assert counts[slot].item() == offset
        assert torch.all(sorted_ids[slot, offset:] == ids.numel())
        assert torch.all(expert_ids[slot, offset // block:] == -1)
