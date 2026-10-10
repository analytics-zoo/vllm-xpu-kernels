# SPDX-License-Identifier: Apache-2.0
"""GPU6-only QSA invisible-partition candidate microbench (not E2E TTFT).

Run OLD and NEW _qwen38_C DSOs in separate processes with identical CLI,
one warm-up and three unprofiled rounds. Optional --profile adds one separate
diagnostic call. ZE_AFFINITY_MASK=6 and explicit approval are mandatory.
No installation or server action is performed.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path


def geometry(rows, length):
    partitions = 32 if length >= 4096 else 1
    max_blocks = (length + 3) // 4
    per_partition = (max_blocks + partitions - 1) // partitions
    block = ((per_partition + 511) // 512) * 512
    return partitions, block, (rows + 127) // 128


def windows_for(rows, length):
    return {"early": 0, "middle": (length - rows) // 2,
            "tail": length - rows}


def empty_fraction(rows, length, start):
    partitions, block, _ = geometry(rows, length)
    empty = 0
    for position in range(start, start + rows):
        visible_blocks = min(position + 1, length) // 4
        empty += sum(partition * block >= visible_blocks
                     for partition in range(partitions))
    total = rows * partitions
    return {"empty_workgroups": empty, "total_workgroups": total,
            "empty_fraction": empty / total,
            "blocks_per_partition": block}


def load_native(path):
    if not path.is_file() or path.suffix != ".so":
        raise ValueError("--library must be an isolated .so")
    spec = importlib.util.spec_from_file_location("_qwen38_C", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if getattr(module, "qsa_sycl_selection_wide_scratch_abi_version", 0) != 1:
        raise RuntimeError("expected identical wide-scratch ABI=1")
    return module


def make_case(torch, rows, length, page_size, start):
    # CPU RNG makes OLD/NEW subprocess inputs byte-identical before transfer.
    torch.manual_seed(71213 + rows + length + page_size)
    pages = (length + 4 * page_size - 1) // (4 * page_size)
    query = torch.randn(rows, 4, 128, dtype=torch.float16).to("xpu:0")
    cache = torch.randn(pages, page_size, 1, 128,
                        dtype=torch.float16).to("xpu:0")
    table = torch.arange(pages, dtype=torch.int32).reshape(1, pages).to(
        "xpu:0")
    requests = torch.zeros(rows, dtype=torch.int32, device="xpu:0")
    positions = torch.arange(start, start + rows, dtype=torch.int64).to(
        "xpu:0")
    seq = torch.tensor([length], dtype=torch.int32, device="xpu:0")
    output = torch.empty(rows, 2051, dtype=torch.int32, device="xpu:0")
    shape = (128, 32, 512)
    scratch = tuple(torch.empty(shape, dtype=dtype, device="xpu:0")
                    for dtype in (torch.float32, torch.int32,
                                  torch.float32, torch.int32))
    return (query, cache, table, requests, positions, seq,
            page_size, length, output, *scratch)


def measure(torch, native, call_args):
    native.select_paged_tokens_v2(*call_args)
    torch.xpu.synchronize()
    rounds = []
    for _ in range(3):
        torch.xpu.synchronize()
        begin = time.perf_counter_ns()
        native.select_paged_tokens_v2(*call_args)
        submitted = time.perf_counter_ns()
        torch.xpu.synchronize()
        completed = time.perf_counter_ns()
        rounds.append({
            "host_call_us": (submitted - begin) / 1000,
            "drain_us": (completed - submitted) / 1000,
            "wall_us": (completed - begin) / 1000,
        })
    return {"rounds": rounds,
            "host_call_median_us": statistics.median(
                item["host_call_us"] for item in rounds),
            "wall_median_us": statistics.median(
                item["wall_us"] for item in rounds)}


def profile_once(torch, native, call_args):
    with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU]) as profiler:
        with torch.profiler.record_function("qsa_visibility_candidate_call"):
            native.select_paged_tokens_v2(*call_args)
        torch.xpu.synchronize()
    events = [event for event in profiler.events()
              if event.device_type == torch.autograd.DeviceType.XPU]
    phase0 = [event for event in events if "SelectionPhase0" in event.name]
    merge = [event for event in events if "SelectionMerge" in event.name]
    return {
        "xpu_event_count": len(events),
        "phase0_count": len(phase0), "merge_count": len(merge),
        "phase0_duration_sum_us_not_wall": sum(
            event.time_range.elapsed_us() for event in phase0),
        "merge_duration_sum_us_not_wall": sum(
            event.time_range.elapsed_us() for event in merge),
        "all_device_duration_sum_us_not_wall": sum(
            event.time_range.elapsed_us() for event in events),
        "device_event_span_us_not_e2e": (
            max(event.time_range.end for event in events) -
            min(event.time_range.start for event in events) if events else 0),
        "event_names": dict(Counter(event.name for event in events)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="new JSONL path; never overwritten")
    parser.add_argument("--rows", nargs="+", type=int,
                        default=[128, 129, 4096])
    parser.add_argument("--lengths", nargs="+", type=int,
                        default=[4096, 65536, 65537, 128000, 256000])
    parser.add_argument("--page-sizes", nargs="+", type=int,
                        default=[128, 64])
    parser.add_argument("--profile", action="store_true",
                        help="separate one-call diagnostic after timed rounds")
    args = parser.parse_args()
    if (os.environ.get("QWEN38_VISIBILITY_GPU6_APPROVED") != "1" or
            os.environ.get("ZE_AFFINITY_MASK") != "6"):
        parser.error("GPU6 approval and ZE_AFFINITY_MASK=6 are required")
    if any(rows not in (128, 129, 4096) for rows in args.rows):
        parser.error("rows must be 128/129/4096")
    if any(length < max(args.rows) or length > 256000
           for length in args.lengths):
        parser.error("each length must be >= max(rows) and <= 256000")
    if any(page not in (64, 128) for page in args.page_sizes):
        parser.error("page sizes must be 64/128")
    library = args.library.resolve(strict=True)
    import torch

    if torch.xpu.device_count() != 1:
        parser.error("GPU6 mask must expose exactly one XPU")
    native = load_native(library)
    with args.output.open("x", encoding="utf-8") as handle:
        def emit(record):
            line = json.dumps(record, ensure_ascii=False)
            handle.write(line + "\n")
            handle.flush()
            print(line, flush=True)

        library_sha = hashlib.sha256(library.read_bytes()).hexdigest()
        emit({"kind": "run", "library": str(library),
              "library_sha256": library_sha,
              "device": torch.xpu.get_device_name(0),
              "physical_mask": "6", "warmups": 1, "measurements": 3,
              "scope": "single-rank selection micro; no E2E TTFT inference"})
        for rows in args.rows:
            for length in args.lengths:
                for page_size in args.page_sizes:
                    for window, start in windows_for(rows, length).items():
                        call_args = make_case(torch, rows, length,
                                              page_size, start)
                        timing = measure(torch, native, call_args)
                        diagnostic = (profile_once(torch, native, call_args)
                                      if args.profile else None)
                        partitions, block, chunks = geometry(rows, length)
                        expected = {"phase0": chunks,
                                    "merge": chunks * 5}
                        if (diagnostic is not None and
                                (diagnostic["phase0_count"] !=
                                 expected["phase0"] or
                                 diagnostic["merge_count"] !=
                                 expected["merge"])):
                            raise AssertionError(
                                "selection path/launch count changed")
                        emit({"kind": "case", "rows": rows,
                              "length": length, "page_size": page_size,
                              "window": window, "position_start": start,
                              "position_end": start + rows - 1,
                              "geometry": {"partitions": partitions,
                                           "blocks_per_partition": block,
                                           "scratch_rows": 128,
                                           "expected_launches": expected},
                              "visibility": empty_fraction(rows, length,
                                                             start),
                              "timing": timing, "profile": diagnostic})


if __name__ == "__main__":
    main()
