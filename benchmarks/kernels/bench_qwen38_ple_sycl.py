"""One warm-up and three measured device-time rounds for TP4 PLE kernels.

This standalone diagnostic never starts a server or installs a package. Run
only after GPU assignment; set ZE_AFFINITY_MASK externally. A final ESIMD
library can be loaded for same-shape comparison with --esimd-library.
"""

import argparse
import json
import statistics
import time
from functools import partial

import torch


def _record(name, call, iterations):
    for _ in range(iterations):
        call()
    torch.xpu.synchronize()
    samples = []
    for _ in range(3):
        begin = time.perf_counter()
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
        ]) as profile:
            for _ in range(iterations):
                call()
            torch.xpu.synchronize()
        wall_us = (time.perf_counter() - begin) * 1e6 / iterations
        kernels = [
            event for event in profile.events()
            if event.device_type == torch.autograd.DeviceType.XPU
            and not event.name.startswith("Memcpy ")
        ]
        if len(kernels) != iterations:
            event_types = sorted(set(event.name for event in kernels))
            raise RuntimeError(f"{name}: expected {iterations} XPU launches; "
                               f"observed {len(kernels)}; "
                               f"event types: {event_types}")
        device_us = sum(event.time_range.elapsed_us() for event in kernels)
        samples.append({
            "device_us": device_us / iterations,
            "wall_us": wall_us
        })
    return {
        "name":
        name,
        "rounds":
        samples,
        "device_median_us":
        statistics.median(item["device_us"] for item in samples),
        "wall_median_us":
        statistics.median(item["wall_us"] for item in samples)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--esimd-library")
    parser.add_argument("--iterations", type=int, default=64)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    torch.ops.load_library(args.library)
    sycl = torch.ops._qwen38_ple_sycl_test
    esimd = None
    if args.esimd_library:
        torch.ops.load_library(args.esimd_library)
        esimd = torch.ops.custom_esimd_kernels_vllm
    if not torch.xpu.is_available():
        raise RuntimeError("XPU unavailable")

    torch.manual_seed(102)
    device = "xpu"
    x = torch.randn((1, 10240), dtype=torch.float16, device=device)
    w = torch.randn((10240, ), dtype=torch.float16, device=device) * 0.1
    q = torch.randn_like(x)
    gate = torch.rand((1, 4), dtype=torch.float16, device=device)
    value = torch.randn((1, 2560), dtype=torch.float16, device=device)
    conv_weight = torch.randn(
        (10240, 4), dtype=torch.float16, device=device) * 0.1
    state = torch.zeros((3, 10240, 16), dtype=torch.float16, device=device)
    slots = torch.tensor([1], dtype=torch.int32, device=device)
    initial = torch.tensor([True], dtype=torch.bool, device=device)
    starts = torch.tensor([0, 1], dtype=torch.int32, device=device)
    accepted = torch.tensor([1], dtype=torch.int32, device=device)
    token_ids = torch.tensor([7], dtype=torch.int64, device=device)
    ngram_starts = torch.tensor([0, 1], dtype=torch.int64, device=device)
    ngram_context = torch.tensor([[11, 12, 13]], dtype=torch.int64,
                                 device=device)
    multipliers = torch.tensor([8191, 65537, 131071, 524287],
                               dtype=torch.int64,
                               device=device)
    vocab_sizes = torch.tensor([19, 19, 23, 23, 29, 29],
                               dtype=torch.int64,
                               device=device)
    offsets = torch.tensor([0, 19, 38, 61, 84, 113],
                           dtype=torch.int64,
                           device=device)
    ngram_output = torch.empty((1, 6), dtype=torch.int64, device=device)
    embedding_weight = torch.randn((256, 64),
                                   dtype=torch.float16,
                                   device=device)
    local_start = torch.tensor([0], dtype=torch.int64, device=device)
    local_rows = torch.tensor([256], dtype=torch.int64, device=device)
    gathered = torch.empty((1, 6 * 64),
                           dtype=torch.float16,
                           device=device)
    results = []
    for backend_name, backend in (("sycl", sycl), ("esimd", esimd)):
        if backend is None:
            continue
        y = torch.empty_like(x)
        g = torch.empty((1, 4), dtype=torch.float16, device=device)
        raw = torch.empty_like(x)
        norm = torch.empty_like(x)
        out = torch.empty_like(x)
        cases = {
            "ngram_ids_m1":
            partial(backend.ple_ngram_ids, token_ids, ngram_starts,
                    ngram_context, multipliers, vocab_sizes, offsets,
                    ngram_output, 0, 2),
            "embedding_gather_m1":
            partial(backend.ple_embedding_gather, ngram_output,
                    embedding_weight, local_start, local_rows, gathered),
            "grouped_norm_m1":
            partial(backend.ple_grouped_norm, x, w, y, 1e-6, 2560),
            "score_gate_m1":
            partial(backend.ple_score_gate, x, q, g, 2560),
            "gated_value_m1":
            partial(backend.ple_gated_value, gate, value, raw, 4),
            "gated_value_norm_m1":
            partial(backend.ple_gated_value_grouped_norm, gate, value, w, raw,
                    norm, 1e-6),
            "residual_add_m1":
            partial(backend.ple_residual_add, x, q, y),
            "conv_decode_m1":
            partial(backend.ple_short_conv_decode_trusted, x, state,
                    conv_weight, slots, initial, out, 4, True, 0),
            "conv_prefill_m1":
            partial(backend.ple_short_conv_prefill_trusted, x, starts, state,
                    conv_weight, slots, initial, out, 4, True, 0),
            "conv_spec_m1":
            partial(backend.ple_short_conv_spec_trusted, x, starts, state,
                    conv_weight, slots, accepted, out, 4, 4, True, 0),
        }
        for rows in (4, 8):
            batch_x = torch.randn((rows, 10240),
                                  dtype=torch.float16,
                                  device=device)
            batch_q = torch.randn_like(batch_x)
            batch_y = torch.empty_like(batch_x)
            batch_g = torch.empty((rows, 4),
                                  dtype=torch.float16,
                                  device=device)
            batch_state = torch.zeros((rows + 2, 10240, 16),
                                      dtype=torch.float16,
                                      device=device)
            batch_slots = torch.arange(1,
                                       rows + 1,
                                       dtype=torch.int32,
                                       device=device)
            batch_initial = torch.ones((rows, ),
                                       dtype=torch.bool,
                                       device=device)
            batch_starts = torch.arange(rows + 1,
                                        dtype=torch.int32,
                                        device=device)
            batch_accepted = torch.full((rows, ),
                                        1,
                                        dtype=torch.int32,
                                        device=device)
            cases.update({
                f"grouped_norm_m{rows}":
                partial(backend.ple_grouped_norm, batch_x, w, batch_y, 1e-6,
                        2560),
                f"score_gate_m{rows}":
                partial(backend.ple_score_gate, batch_x, batch_q, batch_g,
                        2560),
                f"conv_decode_m{rows}":
                partial(backend.ple_short_conv_decode_trusted, batch_x,
                        batch_state, conv_weight, batch_slots, batch_initial,
                        batch_y, 4, True, 0),
                f"conv_spec_m{rows}":
                partial(backend.ple_short_conv_spec_trusted, batch_x,
                        batch_starts, batch_state, conv_weight, batch_slots,
                        batch_accepted, batch_y, 4, 4, True, 0),
            })
        for case_name, call in cases.items():
            result = _record(f"{backend_name}:{case_name}", call,
                             args.iterations)
            print(json.dumps(result), flush=True)
            results.append(result)
    print(json.dumps({"summary": results}), flush=True)


if __name__ == "__main__":
    main()
