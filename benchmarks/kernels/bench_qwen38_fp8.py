# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 TP4/TP8 W8A16: graph timing, hot/rotating weights, three rounds."""

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
from vllm_xpu_kernels import _xpu_C  # noqa: F401


def measure(calls, repeats=24):
    if len(calls) == 1:
        calls = calls * 32
    graph = torch.xpu.XPUGraph()
    for fn in calls:
        fn()
    torch.xpu.synchronize()
    with torch.xpu.graph(graph):
        for fn in calls:
            fn()
    # One warm-up batch, three measured batches. Event intervals include
    # device gaps inside the graph; they are not a sum of profiled kernels.
    for _ in range(repeats):
        graph.replay()
    torch.xpu.synchronize()
    rounds = []
    for _ in range(3):
        start, end = (torch.xpu.Event(enable_timing=True) for _ in range(2))
        begin = time.perf_counter_ns()
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        rounds.append(
            {
                "event_us": start.elapsed_time(end)
                * 1000
                / (repeats * len(calls)),
                "wall_us": (time.perf_counter_ns() - begin)
                / 1000
                / (repeats * len(calls)),
            }
        )
    return {
        "rounds": rounds,
        "event_median_us": statistics.median(r["event_us"] for r in rounds),
        "wall_median_us": statistics.median(r["wall_us"] for r in rounds),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--groups", default="linear,moe")
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location("_qwen38_fp8_C", args.library)
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    torch.set_num_threads(1)
    torch.manual_seed(20261014)
    report = {"torch": torch.__version__, "library": args.library, "cases": []}

    def save(case):
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(case), flush=True)

    if "linear" in args.groups:
        for m in (1, 5):
            for n, k in (
                (12, 2560),
                (24, 2560),
                (2048, 2560),
                (4096, 2560),
                (2560, 768),
                (2560, 1536),
                (640, 2560),
                (2560, 80),
                (2560, 160),
            ):
                ring = [
                    (
                        torch.randn(m, k, device="xpu").half(),
                        (torch.randn(n, k, device="xpu") * 12).to(
                            torch.float8_e4m3fn
                        ),
                        torch.full((1,), 0.0015, device="xpu"),
                    )
                    for _ in range(32)
                ]
                for copies in (1, 32):
                    cases = ring[:copies]
                    custom = [
                        lambda x=x, w=w, s=s: ops.try_qwen38_fp8_linear(x, w, s)
                        for x, w, s in cases
                    ]
                    native = [
                        lambda x=x, w=w, s=s: torch.ops._xpu_C.fp8_gemm_w8a16(
                            x, w.T, s, None
                        )
                        for x, w, s in cases
                    ]
                    item = {
                        "kind": "linear",
                        "m": m,
                        "n": n,
                        "k": k,
                        "copies": copies,
                        "sycl_tla": measure(custom),
                    }
                    try:
                        item["native"] = measure(native)
                    except RuntimeError as e:
                        item["native_error"] = str(e).splitlines()[0]
                    save(item)
    if "moe" in args.groups:
        for i in (80, 160):
            tables = []
            for _ in range(4):

                def fp8(*shape):
                    return (torch.randn(shape, device="xpu") * 12).to(
                        torch.float8_e4m3fn
                    )

                weights = [
                    fp8(512, 2560, 2 * i),
                    torch.full((512,), 0.0015, device="xpu"),
                    fp8(512, i, 2560),
                    torch.full((512,), 0.0015, device="xpu"),
                    fp8(2 * i, 2560),
                    torch.full((1,), 0.0015, device="xpu"),
                    fp8(2560, i),
                    torch.full((1,), 0.0015, device="xpu"),
                    (torch.randn(1, 2560, device="xpu") * 0.02).half(),
                ]
                rw = (torch.randn(512, 2560, device="xpu") * 0.02).half()
                tables.append((weights, rw))
            for m in (1, 5, 8):
                calls = []
                inputs = []
                for c in range(32):
                    w, rw = tables[c % 4]
                    x = torch.randn(m, 2560, device="xpu").half()
                    inputs.append((x, rw, w))
                    workspace = ops.Workspace()
                    calls.append(
                        lambda x=x, w=w, rw=rw, ws=workspace: ws.try_run(
                            x, rw, w
                        )
                    )
                for copies in (1, 32):
                    save(
                        {
                            "kind": "moe",
                            "m": m,
                            "intermediate": i,
                            "copies": copies,
                            "sycl_tla": measure(calls[:copies]),
                        }
                    )
                if args.sweep:
                    for splits in (4, 8, 16):
                        for fuse in (False, True):
                            calls = []
                            for x, rw, w in inputs:
                                ws = ops.Workspace(
                                    up_splits=splits, fuse_down=fuse
                                )
                                calls.append(
                                    lambda x=x, w=w, rw=rw, ws=ws: ws.try_run(
                                        x, rw, w
                                    )
                                )
                            save(
                                {
                                    "kind": "moe_sweep",
                                    "m": m,
                                    "intermediate": i,
                                    "copies": 32,
                                    "splits": splits,
                                    "fuse": fuse,
                                    "sycl_tla": measure(calls),
                                }
                            )


if __name__ == "__main__":
    main()
