# SPDX-License-Identifier: Apache-2.0
"""Build-only MoE output-owner tests; run alone in a fresh DSO process."""

import pytest
import torch

from tests.test_qwen38_moe_provider import _assert_golden

pytest_plugins = ("tests.test_qwen38_moe_provider",)


def _scenario(case, rows: int, offset: int = 0):
    return {
        **case,
        "x": case["x"][offset:offset + rows],
        "logits": case["logits"][offset:offset + rows],
    }


def _workspace(provider, rows: int):
    factory = (provider.get_qwen38_moe_m1_direct_workspace_class()
               if rows == 1
               else provider.get_qwen38_moe_multi_direct_workspace_class())
    return factory()


def _run(workspace, scenario, grouped=False):
    args = (scenario["x"], scenario["router"], scenario["router_scale"],
            scenario["weights"], scenario["k"])
    if scenario["x"].shape[0] > 1:
        args += (grouped,)
    output = workspace.try_run(*args)
    assert output is not None
    return output


@pytest.mark.parametrize("rows", (1, 5))
def test_retained_outputs_and_third_ephemeral_match_independent_math(
        provider, case, rows):
    workspace = _workspace(provider, rows)
    scenarios = [_scenario(case, rows, offset) for offset in (0, 1, 2, 3)]
    first, second, third = (_run(workspace, scenario)
                            for scenario in scenarios[:3])
    pointers = [output.data_ptr() for output in (first, second, third)]
    assert len(set(pointers)) == 3  # Two bounded slots, then overflow.
    snapshots = [output.cpu().clone() for output in (first, second, third)]
    for output, scenario in zip((first, second, third), scenarios):
        _assert_golden(output, scenario, rows)

    del first  # The other slot and overflow stay live.
    fourth = _run(workspace, scenarios[3])
    assert fourth.data_ptr() == pointers[0]
    _assert_golden(fourth, scenarios[3], rows)
    for output, snapshot in zip((second, third), snapshots[1:]):
        torch.testing.assert_close(output.cpu(), snapshot, atol=0, rtol=0)


@pytest.mark.parametrize("rows", (1, 3, 7))
def test_storage_view_and_live_input_prevent_slot_reuse(provider, case, rows):
    workspace = _workspace(provider, rows)
    first = _run(workspace, _scenario(case, rows, 0))
    storage = first.untyped_storage()
    view = torch.empty(0, dtype=first.dtype, device=first.device)
    view.set_(storage, 0, first.size(), first.stride())
    assert view._base is None  # Independent TensorImpl, shared StorageImpl.
    snapshot = view.cpu().clone()
    old_pointer = first.data_ptr()
    del storage, first

    # No live input aliases the old output here: only StorageImpl ownership
    # must prevent overwriting the independent view.
    second_case = _scenario(case, rows, 1)
    second = _run(workspace, second_case)
    assert second.data_ptr() != old_pointer
    _assert_golden(second, second_case, rows)

    # Now the view is a live input, while the second slot is still held.
    args = (view, case["router"], case["router_scale"], case["weights"],
            case["k"])
    if rows > 1:
        args += (False,)
    third = workspace.try_run(*args)
    assert third is not None
    assert third.data_ptr() not in (old_pointer, second.data_ptr())
    assert not torch._C._is_alias_of(third, view)
    torch.testing.assert_close(view.cpu(), snapshot, atol=0, rtol=0,
                               equal_nan=True)


def test_mutated_return_metadata_does_not_change_private_slot(provider, case):
    workspace = _workspace(provider, 1)
    first = _run(workspace, _scenario(case, 1))
    first_pointer = first.data_ptr()
    assert first._base is None
    torch.xpu.synchronize()
    first.as_strided_((1, 2559), (2560, 1))
    del first
    next_case = _scenario(case, 1, 1)
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=False) as profiler:
        second = _run(workspace, next_case)
    assert second.shape == (1, 2560)
    assert second.data_ptr() == first_pointer
    assert sum(event.name == "aten::empty" for event in profiler.events()) == 0
    _assert_golden(second, next_case, 1)


@pytest.mark.parametrize("rows,grouped", ((2, False), (3, True),
                                          (5, False), (8, True)))
def test_ragged_multi_rows_keep_prior_result(provider, case, rows, grouped):
    workspace = _workspace(provider, rows)
    first_case = _scenario(case, rows, 0)
    second_case = _scenario(case, rows, 1)
    first = _run(workspace, first_case, grouped)
    snapshot = first.cpu().clone()
    second = _run(workspace, second_case, grouped)
    assert first.data_ptr() != second.data_ptr()
    _assert_golden(first, first_case, rows)
    _assert_golden(second, second_case, rows)
    torch.testing.assert_close(first.cpu(), snapshot, atol=0, rtol=0)


@pytest.mark.parametrize("rows", (1, 5))
def test_released_slots_reuse_without_aten_empty(provider, case, rows):
    workspace = _workspace(provider, rows)
    args = _scenario(case, rows, 0)
    first = _run(workspace, args)
    second = _run(workspace, args)
    pointers = {first.data_ptr(), second.data_ptr()}
    assert len(pointers) == 2
    del first, second
    torch.xpu.synchronize()

    # The only profiled call is the direct owner entry; pre-created inputs
    # and a warm two-slot scratch must not allocate a new output Tensor.
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=False) as profiler:
        reused = _run(workspace, args)
    assert reused.data_ptr() in pointers
    assert sum(event.name == "aten::empty" for event in profiler.events()) == 0
    _assert_golden(reused, args, rows)


@pytest.mark.parametrize("rows", (1, 5))
def test_two_nondefault_streams_allow_early_release(provider, case, rows):
    workspace = _workspace(provider, rows)
    scenarios = [_scenario(case, rows, offset) for offset in (0, 1, 2, 3)]
    torch.xpu.synchronize()  # All inputs were prepared on the default stream.
    first_stream, second_stream = torch.xpu.Stream(), torch.xpu.Stream()

    with torch.xpu.stream(first_stream):
        discarded = _run(workspace, scenarios[0])
        first_pointer = discarded.data_ptr()
        del discarded  # The first result has no consumer, but is still queued.
        first_live = _run(workspace, scenarios[1])
        assert first_live.data_ptr() == first_pointer
    with torch.xpu.stream(second_stream):
        discarded = _run(workspace, scenarios[2])
        second_pointer = discarded.data_ptr()
        del discarded
        second_live = _run(workspace, scenarios[3])
        assert second_live.data_ptr() == second_pointer

    torch.xpu.current_stream().wait_stream(first_stream)
    torch.xpu.current_stream().wait_stream(second_stream)
    assert first_live.data_ptr() != second_live.data_ptr()
    _assert_golden(first_live, scenarios[1], rows)
    _assert_golden(second_live, scenarios[3], rows)


@pytest.mark.parametrize("rows", (1, 9))
def test_public_run_out_remains_caller_owned(provider, case, rows):
    operation = ((provider.compact80_ops() if case["k"] == 80
                  else provider.compact160_ops())[0 if rows == 1 else 1])
    scenario = _scenario(case, rows)
    caller_output = torch.full((rows, 2560), -7, dtype=torch.float16,
                               device="xpu")
    returned = operation(scenario["x"], scenario["logits"],
                         *case["weights"], caller_output, 10, 1, 512)
    assert returned.data_ptr() == caller_output.data_ptr()
    _assert_golden(caller_output, scenario, rows)
