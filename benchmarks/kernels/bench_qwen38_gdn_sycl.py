# SPDX-License-Identifier: Apache-2.0
"""GDN SYCL vs installed ESIMD: one warm-up batch, three measured batches.

Only use this after a GPU has been released to the task. Device time sums
visible kernels, whereas wall time also includes dispatcher and queue gaps.
The standalone library is build-only; this script never installs anything.
"""

import argparse
from collections import defaultdict
import json
import statistics
import time

import torch


def measure(fn, ring, iterations):
    for i in range(iterations):
        fn(*ring[i % len(ring)])
    torch.xpu.synchronize()
    unprofiled_rounds = []
    for _ in range(3):
        start = time.perf_counter_ns()
        for i in range(iterations):
            fn(*ring[i % len(ring)])
        submitted = time.perf_counter_ns()
        torch.xpu.synchronize()
        drained = time.perf_counter_ns()
        unprofiled_rounds.append({
            "host_us": (submitted - start) / 1000 / iterations,
            "wall_us": (drained - start) / 1000 / iterations,
        })
    rounds = []
    names = set()
    by_kernel = defaultdict(list)
    for _ in range(3):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        ) as profile:
            for i in range(iterations):
                fn(*ring[i % len(ring)])
            torch.xpu.synchronize()
        events = [
            e
            for e in profile.events()
            if e.device_type == torch.autograd.DeviceType.XPU
        ]
        names.update(e.name for e in events)
        kernel_round = defaultdict(float)
        for event in events:
            name = event.name
            if (
                "GdnConvKernel" in name
                or "GdnConvPackedParallelKernel" in name
                or "gdn_spec_v2_conv_host" in name
            ):
                family = "conv"
            elif (
                "GdnRecurrentKernel" in name
                or "gdn_spec_v2_recurrent_host" in name
            ):
                family = "recurrent"
            elif "GdnNormGateKernel" in name or "rms_norm_gated" in name:
                family = "norm_gate"
            else:
                family = "other"
            kernel_round[family] += event.time_range.elapsed_us() / iterations
        for family, duration in kernel_round.items():
            by_kernel[family].append(duration)
        rounds.append(
            {
                "device_us": sum(e.time_range.elapsed_us() for e in events)
                / iterations,
                "kernel_count": len(events),
            }
        )
    return {
        "rounds": rounds,
        "unprofiled_rounds": unprofiled_rounds,
        "device_median_us": statistics.median(x["device_us"] for x in rounds),
        "host_median_us": statistics.median(
            x["host_us"] for x in unprofiled_rounds),
        "wall_median_us": statistics.median(
            x["wall_us"] for x in unprofiled_rounds),
        "kernels": sorted(names),
        "phase_median_us": {
            family: statistics.median(times)
            for family, times in by_kernel.items()
        },
    }


def make_case(kind, m, slots=12):
    h, hv = 4, 12
    dim = (2 * h + hv) * 128
    qkvz = (torch.randn(m, dim + hv * 128, device="xpu") * 0.06).half()
    conv_len = m + 2 if kind == "spec" else 3
    conv = (torch.randn(slots, conv_len, dim, device="xpu") * 0.02).half()
    weight = (torch.randn(dim, 4, device="xpu") * 0.1).half()
    bias = torch.zeros(dim, dtype=torch.float16, device="xpu")
    idx = (
        torch.arange(m, dtype=torch.int32, device="xpu") + 1
        if kind == "spec"
        else torch.arange(m, dtype=torch.int32, device="xpu")
    )
    a_log = torch.full(
        (hv,),
        0.2,
        dtype=torch.float32 if kind == "spec" else torch.float16,
        device="xpu",
    )
    dt_bias = torch.zeros(hv, dtype=torch.float16, device="xpu")
    ba = (torch.randn(m, 2 * hv, device="xpu") * 0.05).half()
    ssm = (torch.randn(slots, hv, 128, 128, device="xpu") * 0.001).half()
    output = torch.empty(m, hv, 128, dtype=torch.float16, device="xpu")
    z = torch.empty_like(output)
    common = [qkvz, conv, weight, bias, idx, a_log, dt_bias, ba, ssm]
    if kind == "decode":
        return (*common, idx, output, z, 1 / 128**0.5)
    token_idx = torch.arange(m, dtype=torch.int32, device="xpu")
    accepted = torch.ones(1, dtype=torch.int32, device="xpu")
    return (*common, output, z, token_idx, accepted, 1, m, 1 / 128**0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--ring", type=int, default=8)
    parser.add_argument(
        "--kind", choices=["decode", "spec", "norm", "all"], default="all"
    )
    parser.add_argument("--no-esimd", action="store_true")
    args = parser.parse_args()
    torch.ops.load_library(args.library)
    ops = torch.ops.qwen38_gdn_sycl_test
    esimd = None
    if not args.no_esimd:
        import custom_esimd_kernels_vllm as esimd
    torch.manual_seed(834)
    kinds = (
        ("decode", 1),
        ("decode", 2),
        ("decode", 4),
        ("decode", 8),
        ("spec", 2),
        ("spec", 4),
        ("spec", 8),
    )
    for kind, m in kinds:
        if args.kind not in ("all", kind):
            continue
        ring = [make_case(kind, m) for _ in range(args.ring)]
        old_ring = None
        if esimd is not None:
            # Snapshot before native in-place warm-up or measurement, not after.
            old_ring = [tuple(value.clone() if isinstance(value, torch.Tensor)
                              else value for value in case) for case in ring]
        new_fn = ops.decode if kind == "decode" else ops.spec_v2
        item = {
            "kind": kind,
            "m": m,
            "tp": 4,
            "h": 4,
            "hv": 12,
            "ring": args.ring,
            "iterations": args.iterations,
            "sycl": measure(new_fn, ring, args.iterations),
        }
        if esimd is not None:
            old_fn = (
                esimd.esimd_gdn_conv_fused_seq
                if kind == "decode"
                else esimd.esimd_gdn_conv_fused_seq_spec_v2
            )

            # The ESIMD API takes explicit H/HV/K/V; the SYCL test ABI infers
            # these from validated tensor shapes. Keep all other inputs same.
            def old_call(*case, mode=kind, fn=old_fn, rows=m):
                if mode == "decode":
                    fn(*case[:-1], rows, 4, 12, 128, 128, case[-1])
                else:
                    fn(*case[:-1], 4, 12, 128, 128, case[-1])

            item["esimd"] = measure(old_call, old_ring, args.iterations)
            item["device_ratio"] = (
                item["sycl"]["device_median_us"]
                / item["esimd"]["device_median_us"]
            )
        print(json.dumps(item), flush=True)

    if args.kind in ("all", "norm"):
        ring = []
        old_ring = []
        for _ in range(args.ring):
            x = (torch.randn(1, 12, 128, device="xpu") * 0.08).half()
            z = (torch.randn_like(x) * 0.08).half()
            w = torch.ones(128, dtype=torch.float16, device="xpu")
            y = torch.empty(1, 12 * 128, dtype=torch.float16, device="xpu")
            ring.append((x, z, w, y, 1e-6, False))
            old_ring.append(
                (
                    x.view(12, 128),
                    z.view(12, 128),
                    w,
                    torch.empty_like(x).view(12, 128),
                    1e-6,
                )
            )
        item = {
            "kind": "norm",
            "m": 1,
            "tp": 4,
            "sycl": measure(ops.norm_gate, ring, args.iterations),
        }
        if esimd is not None:
            item["esimd"] = measure(
                esimd.esimd_rms_norm_gated, old_ring, args.iterations
            )
            item["device_ratio"] = (
                item["sycl"]["device_median_us"]
                / item["esimd"]["device_median_us"]
            )
        print(json.dumps(item), flush=True)


if __name__ == "__main__":
    main()
