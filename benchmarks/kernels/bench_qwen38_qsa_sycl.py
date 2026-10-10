# SPDX-License-Identifier: Apache-2.0
"""Isolated QSA SYCL vs installed ESIMD, one warm-up and three measurements.

These are device-kernel and host+submit timings, not E2E model claims. Run
under ZE_AFFINITY_MASK=4 and pass --library to the isolated partial build.
"""

import argparse
import importlib
import importlib.util
import json
import statistics
import sys
import time

import torch


def load_native(path):
    spec = importlib.util.spec_from_file_location("qwen38_qsa_sycl_dev", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def measure(fn, iterations):
    for _ in range(iterations):
        fn()
    torch.xpu.synchronize()
    rounds = []
    names = set()
    for _ in range(3):
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
        ]) as profiler:
            begin = time.perf_counter_ns()
            for _ in range(iterations):
                fn()
            torch.xpu.synchronize()
            wall_us = (time.perf_counter_ns() - begin) / iterations / 1000
        events = [
            e for e in profiler.events()
            if e.device_type == torch.autograd.DeviceType.XPU
        ]
        names.update(e.name for e in events)
        per_kernel = {}
        for event in events:
            per_kernel[event.name] = (per_kernel.get(event.name, 0.0) +
                                      event.time_range.elapsed_us() /
                                      iterations)
        rounds.append({
            "device_us":
            sum(e.time_range.elapsed_us() for e in events) / iterations,
            "wall_us":
            wall_us,
            "kernels_per_call":
            len(events) / iterations,
            "per_kernel_us":
            per_kernel,
        })
    return {
        "device_median_us":
        statistics.median(r["device_us"] for r in rounds),
        "wall_median_us":
        statistics.median(r["wall_us"] for r in rounds),
        "rounds":
        rounds,
        "kernel_names":
        sorted(names),
    }


def attention_case(rows, length, page_size, heads):
    torch.manual_seed(713 + rows + length)
    pages = (length + page_size - 1) // page_size
    q = torch.randn(rows, heads, 256, device="xpu", dtype=torch.float16)
    packed = torch.randn(pages, 1, page_size, 512, device="xpu",
                         dtype=torch.float16)
    indices = torch.full((rows, 2051), -1, device="xpu",
                         dtype=torch.int32)
    if length >= 2051:
        for row in range(rows):
            indices[row] = torch.randperm(length, device="xpu")[:2051].int()
    else:
        for row in range(rows):
            indices[row, :length] = torch.arange(length, device="xpu",
                                                 dtype=torch.int32)
    table = torch.arange(pages, device="xpu", dtype=torch.int32).reshape(
        1, pages)
    reqs = torch.zeros(rows, device="xpu", dtype=torch.int32)
    output = torch.empty_like(q)
    partials = torch.empty(rows, heads, 43, 258, device="xpu",
                           dtype=torch.float32)
    return (q, packed, indices, table, reqs, page_size, output, partials)


def compression_case(rows):
    torch.manual_seed(121)
    raw = torch.randn(rows, 1, 128, device="xpu", dtype=torch.float16)
    pos = torch.randint(0, 1000, (rows, 1, 3), device="xpu",
                        dtype=torch.int64)
    ring = torch.randn(1, 8, 1, 128, device="xpu", dtype=torch.float16)
    ring_pos = torch.randint(0, 1000, (1, 8, 1, 3), device="xpu",
                             dtype=torch.int64)
    table = torch.zeros((1, 1), device="xpu", dtype=torch.int32)
    reqs = torch.zeros(rows, device="xpu", dtype=torch.int32)
    starts = torch.tensor([0, rows], device="xpu", dtype=torch.int32)
    logical = torch.arange(1000, 1000 + rows, device="xpu",
                           dtype=torch.int64)
    slots = torch.arange(rows, device="xpu", dtype=torch.int64)
    pooled = torch.empty_like(raw)
    first = torch.empty((rows, 3), device="xpu", dtype=torch.int64)
    return (raw, pos, ring, ring_pos, table, reqs, starts, logical, slots,
            pooled, first, 4, 2048, True)


def selection_case(rows, length, page_size=128, scratch_rows=32):
    torch.manual_seed(307 + rows + length)
    pages = (length + 4 * page_size - 1) // (4 * page_size)
    q = torch.randn(rows, 4, 128, dtype=torch.float16, device="xpu")
    cache = torch.randn(pages, page_size, 1, 128,
                        dtype=torch.float16, device="xpu")
    table = torch.arange(pages, device="xpu", dtype=torch.int32).reshape(
        1, pages)
    reqs = torch.zeros(rows, device="xpu", dtype=torch.int32)
    positions = torch.full((rows, ), length - 1, device="xpu",
                           dtype=torch.int64)
    seq = torch.tensor([length], device="xpu", dtype=torch.int32)
    out = torch.empty(rows, 2051, device="xpu", dtype=torch.int32)
    work_rows = min(rows, scratch_rows)
    scores_a = torch.empty(work_rows, 32, 512, device="xpu",
                           dtype=torch.float32)
    indices_a = torch.empty(work_rows, 32, 512, device="xpu",
                            dtype=torch.int32)
    scores_b = torch.empty_like(scores_a)
    indices_b = torch.empty_like(indices_a)
    return (q, cache, table, reqs, positions, seq, page_size, length,
            out, scores_a, indices_a, scores_b, indices_b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--iterations", type=int, default=48)
    parser.add_argument("--case", choices=["all", "attention", "compression",
                                           "selection"],
                        default="all")
    args = parser.parse_args()
    native = load_native(args.library)
    esimd = importlib.import_module("custom_esimd_kernels_vllm.qsa_ops")
    report = {
        "torch": torch.__version__,
        "sycl_library": args.library,
        "esimd_library": esimd.__file__,
        "iterations_per_round": args.iterations,
        "warmup_rounds": 1,
        "measured_rounds": 3,
        "cases": [],
    }
    if args.case in ("all", "compression"):
        for rows in (1, 2, 4, 8):
            case = compression_case(rows)
            native_call = lambda case=case: native.group_compress_v2(*case)
            esimd_call = lambda case=case: esimd.qsa_group_compress_v2(*case)
            result = {
                "case": "group_compression",
                "rows": rows,
                "sycl": measure(native_call, args.iterations),
                "esimd": measure(esimd_call, args.iterations),
            }
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
    if args.case in ("all", "attention"):
        for rows, length, page_size in ((1, 1024, 256), (4, 1024, 256),
                                        (8, 1024, 256), (1, 131072, 512),
                                        (4, 131072, 512)):
            case = attention_case(rows, length, page_size, heads=6)
            native_call = lambda case=case: native.token_split_attention_v3(
                *case)
            esimd_call = lambda case=case: (
                esimd.sparse_attention_token_split_candidate_q6_v1(*case))
            result = {
                "case": "token_split_attention",
                "rows": rows,
                "length": length,
                "page_size": page_size,
                "sycl": measure(native_call, args.iterations),
                "esimd": measure(esimd_call, args.iterations),
            }
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
    if args.case in ("all", "selection"):
        wide_capable = (getattr(
            native, "qsa_sycl_selection_wide_scratch_abi_version", 0) == 1)
        for rows, length in ((1, 1024), (1, 131072), (4, 131072),
                             (128, 4096), (129, 131072)):
            scratch_options = ((32, 128) if wide_capable and rows >= 128
                               and length >= 4096 else (32, ))
            for scratch_rows in scratch_options:
                case = selection_case(rows, length,
                                      scratch_rows=scratch_rows)
                native_call = lambda case=case: native.select_paged_tokens_v2(
                    *case)
                reference_op = (esimd.qsa_select_paged_tokens_local_v1
                                if length < 4096 else
                                esimd.qsa_select_paged_tokens_parallel_v1
                                if rows <= 64 else
                                esimd.qsa_select_paged_tokens_v2)
                esimd_call = lambda case=case, reference_op=reference_op: (
                    reference_op(*case[:6], 2048, 4, 128, case[8]))
                result = {
                    "case": "selection",
                    "rows": rows,
                    "length": length,
                    "scratch_rows": scratch_rows,
                    "sycl": measure(native_call, args.iterations),
                    "esimd": measure(esimd_call, args.iterations),
                }
                report["cases"].append(result)
                print(json.dumps(result), flush=True)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
