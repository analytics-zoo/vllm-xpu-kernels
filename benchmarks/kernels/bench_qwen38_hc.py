"""Same-process ordinary-SYCL/installed-ESIMD HC kernel comparison.

One warm-up batch and three measured batches; device durations are sums of
profiler XPU kernels per invocation, not E2E time. The installed ESIMD DSO is
an informative reference but may lag its source commit.
"""

import argparse
import json
import statistics
import time

import torch


def measure(fn, ring, iterations):
    for index in range(iterations):
        fn(*ring[index % len(ring)])
    torch.xpu.synchronize()
    rounds = []
    names = set()
    for _ in range(3):
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
        ]) as profiler:
            started = time.perf_counter_ns()
            for index in range(iterations):
                fn(*ring[index % len(ring)])
            torch.xpu.synchronize()
            wall_us = (time.perf_counter_ns() - started) / iterations / 1000
        kernels = [
            event for event in profiler.events()
            if event.device_type == torch.autograd.DeviceType.XPU
        ]
        names.update(event.name for event in kernels)
        rounds.append({
            "device_us":
            sum(event.time_range.elapsed_us()
                for event in kernels) / iterations,
            "wall_us":
            wall_us,
            "kernel_count":
            len(kernels),
        })
    return {
        "rounds":
        rounds,
        "median_device_us":
        statistics.median(result["device_us"] for result in rounds),
        "median_wall_us":
        statistics.median(result["wall_us"] for result in rounds),
        "kernel_names":
        sorted(names),
    }


def measure_wall_only(fn, ring, iterations):
    for index in range(iterations):
        fn(*ring[index % len(ring)])
    torch.xpu.synchronize()
    rounds = []
    for _ in range(3):
        started = time.perf_counter_ns()
        for index in range(iterations):
            fn(*ring[index % len(ring)])
        torch.xpu.synchronize()
        rounds.append((time.perf_counter_ns() - started) / iterations / 1000)
    return {"rounds_us": rounds, "median_us": statistics.median(rounds)}


def make_case(case, m, ring_size):
    torch.manual_seed(2157)
    ring = []
    for _ in range(ring_size):
        x = torch.randn(m, 10240, device="xpu", dtype=torch.float16) * 0.2
        block = torch.randn(m, 2560, device="xpu", dtype=torch.float16)
        inject = torch.randn(m, 4, device="xpu", dtype=torch.float16)
        norm_weight = torch.randn(10240, device="xpu", dtype=torch.float16)
        down_weight = torch.randn(
            336, 10240, device="xpu", dtype=torch.float16) / 32
        up_weight = torch.randn(10240, 320, device="xpu",
                                dtype=torch.float16) / 20
        combined = torch.empty_like(x)
        normed = torch.empty_like(x)
        low = torch.empty(m, 336, device="xpu", dtype=torch.float16)
        gate = torch.empty_like(x)
        mixed = torch.empty_like(block)
        if case == "combine_norm":
            ring.append(
                (x, block, inject, norm_weight, combined, normed, 1e-6))
        elif case == "down":
            ring.append((x, down_weight, low))
        elif case == "up_gate_mix":
            low.normal_()
            ring.append((low[:, :320], up_weight, x, mixed))
        else:
            ring.append((
                x,
                block,
                inject,
                norm_weight,
                down_weight,
                up_weight,
                combined,
                normed,
                low,
                gate,
                mixed,
                1e-6,
            ))
    return ring


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--reference-library")
    parser.add_argument("--case",
                        choices=[
                            "all", "combine_norm", "down", "up_gate_mix",
                            "combine_mix", "workspace"
                        ],
                        default="all")
    parser.add_argument("--m", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--ring", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--iterations", type=int, default=96)
    args = parser.parse_args()
    torch.ops.load_library(args.library)
    import custom_esimd_kernels_vllm.custom_esimd_kernels  # noqa: F401
    if args.reference_library:
        torch.ops.load_library(args.reference_library)

    native = torch.ops.qwen38_hc_sycl
    previous = torch.ops.custom_esimd_kernels_vllm
    final_reference = (torch.ops.qwen38_hc_final_esimd_reference
                       if args.reference_library else None)
    cases = ([
        "combine_norm", "down", "up_gate_mix", "combine_mix", "workspace"
    ] if args.case == "all" else [args.case])
    for case in cases:
        for m in args.m:
            if case == "combine_mix" and m != 1:
                continue
            if case == "combine_norm":
                sycl_op = native.combine_norm
                esimd_op = (previous.hc_combine_norm_v1
                            if m == 1 else previous.hc_combine_norm_multi_m_v1)
                if final_reference is not None and m == 1:
                    esimd_op = final_reference.combine_norm
            elif case == "down":
                sycl_op = native.down
                esimd_op = (previous.esimd_hc_down_fp16_out if m == 1 else
                            previous.esimd_hc_down_fp16_multi_m_out_v1)
                if final_reference is not None:
                    esimd_op = final_reference.down
            elif case == "up_gate_mix":
                sycl_op = native.up_gate_mix
                esimd_op = (previous.esimd_hc_up_gate_mix_m1_v1 if m == 1 else
                            previous.esimd_hc_up_gate_mix_multi_m_v1)
                if final_reference is not None:
                    esimd_op = final_reference.up_gate_mix
            elif case == "workspace":
                new_class = (torch.classes.qwen38_hc_sycl.HCWorkspace
                             if m == 1 else
                             torch.classes.qwen38_hc_sycl.HCMultiMWorkspaceV1)
                old_class = (
                    torch.classes.custom_esimd_kernels_vllm.HCWorkspace
                    if m == 1 else
                    torch.classes.custom_esimd_kernels_vllm.HCMultiMWorkspaceV1
                )
                sycl_owner = new_class()
                esimd_owner = old_class()
                sycl_op = sycl_owner.try_run
                # Installed ESIMD Multi class predates its optional direct
                # try_run entry, but its final host-chain run ABI is stable.
                esimd_op = (esimd_owner.try_run if m == 1 else esimd_owner.run)
            else:

                def sycl_op(hidden, block, injection, norm_weight, down_weight,
                            up_weight, combined, normed, down, gate, mixed,
                            eps):
                    native.combine_mix(
                        hidden,
                        block,
                        injection,
                        norm_weight,
                        down_weight,
                        up_weight,
                        combined,
                        normed,
                        down,
                        mixed,
                        eps,
                    )

                esimd_op = previous.hc_combine_mix_m1_v2
            for ring_size in args.ring:
                full_ring = make_case(case, m, ring_size)
                ring = full_ring
                if case == "workspace":
                    ring = [(*row[:6], row[-1]) for row in ring]
                esimd_op(*ring[0])
                result = {
                    "case": case,
                    "m": m,
                    "ring": ring_size,
                    "iterations": args.iterations,
                    "esimd": measure(esimd_op, ring, args.iterations),
                    "sycl": measure(sycl_op, ring, args.iterations),
                }
                result["ratio_device_sycl_over_esimd"] = (
                    result["sycl"]["median_device_us"] /
                    result["esimd"]["median_device_us"])
                if case == "workspace":

                    def python_three_ops(hidden, block, injection, norm_weight,
                                         down_weight, up_weight, combined,
                                         normed, down, gate, mixed, eps):
                        native.combine_norm(hidden, block, injection,
                                            norm_weight, combined, normed, eps)
                        native.down(normed, down_weight, down)
                        native.up_gate_mix(down[:, :320], up_weight, normed,
                                           mixed)

                    result["wall_only_sycl_workspace"] = measure_wall_only(
                        sycl_op, ring, args.iterations)
                    result["wall_only_sycl_python_three_ops"] = (
                        measure_wall_only(python_three_ops, full_ring,
                                          args.iterations))
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
