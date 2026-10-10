# SPDX-License-Identifier: Apache-2.0
"""Same-process ESIMD/SYCL A/B, with hot and rotating working sets.

One warm-up batch and three profiled batches per provider. Device kernel
durations exclude host queue gaps; wall time includes Python/dispatcher and
draining the stream. Profiling can perturb both measurements.
"""

import argparse
import json
import statistics
import time

import torch


def measure(fn, ring, iterations):
    for i in range(iterations):
        fn(*ring[i % len(ring)])
    torch.xpu.synchronize()
    results = []
    names = set()
    for _ in range(3):
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
        ]) as profiler:
            start = time.perf_counter_ns()
            for i in range(iterations):
                fn(*ring[i % len(ring)])
            torch.xpu.synchronize()
            wall_us = (time.perf_counter_ns() - start) / 1000 / iterations
        events = [
            e for e in profiler.events()
            if e.device_type == torch.autograd.DeviceType.XPU
        ]
        names.update(e.name for e in events)
        results.append({
            "device_us":
            sum(e.time_range.elapsed_us() for e in events) / iterations,
            "wall_us":
            wall_us,
            "kernel_count":
            len(events),
        })
    return {
        "rounds": results,
        "device_median_us": statistics.median(r["device_us"] for r in results),
        "wall_median_us": statistics.median(r["wall_us"] for r in results),
        "kernel_names": sorted(names),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--reference-library")
    parser.add_argument("--iterations", type=int, default=96)
    parser.add_argument("--ring", type=int, default=48)
    parser.add_argument("--m", type=int, help="Restrict the token count")
    parser.add_argument("--shape",
                        choices=["all", "fused", "single"],
                        default="all")
    args = parser.parse_args()
    torch.ops.load_library(args.library)
    import custom_esimd_kernels_vllm as esimd

    native = torch.ops._qwen38_C
    if args.reference_library:
        torch.ops.load_library(args.reference_library)
    torch.manual_seed(47)
    report = {
        "torch": torch.__version__,
        "library": args.library,
        "iterations": args.iterations,
        "cases": []
    }
    cases = []
    if args.shape in ("all", "fused"):
        cases.append(("gdn_input_fused2", 1, 4096, 2560, 24))
    if args.shape in ("all", "single"):
        cases.extend(
            ("linear", m, n, k, 0)
            for m, n, k in [(1, 4096, 2560), (1, 2560, 1536), (
                1, 2560, 2560), (2, 4096, 2560), (5, 4096,
                                                  2560), (8, 4096,
                                                          2560), (5, 24,
                                                                  2560)])
    if args.m is not None:
        cases = [case for case in cases if case[1] == args.m]
    for name, m, n, k, n1 in cases:
        for ring_size in (1, args.ring):
            ring = []
            for _ in range(ring_size):
                x = torch.randn(m, k, dtype=torch.float16, device="xpu")
                w = torch.randint(256, (n, k // 2),
                                  dtype=torch.uint8,
                                  device="xpu")
                s = (torch.randn(n, k // 128) * 0.02).half().to("xpu")
                out = torch.empty(m, n, dtype=torch.float16, device="xpu")
                tensors = (x, w, s, out)
                if n1:
                    w1 = torch.randint(256, (n1, k // 2),
                                       dtype=torch.uint8,
                                       device="xpu")
                    s1 = (torch.randn(n1, k // 128) * 0.02).half().to("xpu")
                    out1 = torch.empty(m,
                                       n1,
                                       dtype=torch.float16,
                                       device="xpu")
                    tensors += (w1, s1, out1)
                ring.append(tensors)
            if n1:
                old_fn = esimd.esimd_gemv_int4_fused2
                if args.reference_library:
                    old_fn = torch.ops.qwen38_esimd_reference.fused2
                new_fn = native.int4_linear_fused2
            elif m == 1:
                old_fn = esimd.esimd_gemv_int4
                if args.reference_library:
                    old_fn = torch.ops.qwen38_esimd_reference.gemv
                new_fn = native.int4_linear
            elif n == 24:
                old_fn = esimd.esimd_gemm_int4_small_n_v1
                new_fn = native.int4_linear
            else:
                old_fn = esimd.esimd_gemm_int4_pgrp
                new_fn = native.int4_linear
            old_fn(*ring[0])
            reference = ring[0][3].clone()
            reference1 = ring[0][6].clone() if n1 else None
            new_fn(*ring[0])
            difference = (ring[0][3] - reference).abs().float()
            item = {
                "name": name,
                "m": m,
                "n": n,
                "k": k,
                "n1": n1,
                "ring": ring_size,
                "max_abs_diff": difference.max().item(),
                "rms_diff": difference.square().mean().sqrt().item()
            }
            if reference1 is not None:
                item["max_abs_diff1"] = (ring[0][6] -
                                         reference1).abs().max().item()
            item["esimd"] = measure(old_fn, ring, args.iterations)
            item["sycl"] = measure(new_fn, ring, args.iterations)
            item["device_ratio_sycl_over_esimd"] = (
                item["sycl"]["device_median_us"] /
                item["esimd"]["device_median_us"])
            report["cases"].append(item)
            print(json.dumps(item), flush=True)
            del ring, tensors, x, w, s, out
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
