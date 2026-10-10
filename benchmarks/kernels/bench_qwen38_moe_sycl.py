# SPDX-License-Identifier: Apache-2.0
"""Standalone compact DOWN comparison; run only after the sidecar is linked."""

import argparse
import statistics
import time

import torch


def measure(op, x, weight, scale, counts, repeats: int) -> float:
    # One warm-up, three measurements. Includes launch/host overhead, excludes
    # import/model setup. Synchronization is outside each measurement loop.
    op(x, weight, scale, counts)
    torch.xpu.synchronize()
    times = []
    for _ in range(3):
        torch.xpu.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            op(x, weight, scale, counts)
        torch.xpu.synchronize()
        times.append(1e3 * (time.perf_counter() - start) / repeats)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, choices=(80, 160), required=True)
    parser.add_argument("--m", type=int, required=True,
                        help="expert-sorted route rows, not input tokens")
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--new-lib", help="path to sidecar extension .so")
    parser.add_argument("--old-lib", help="path to ESIMD reference .so")
    args = parser.parse_args()
    if not torch.xpu.is_available():
        raise RuntimeError("XPU is required")
    if args.m < 1 or args.repeats < 1:
        raise ValueError("m and repeats must be positive")
    if args.new_lib:
        torch.ops.load_library(args.new_lib)
    if args.old_lib:
        torch.ops.load_library(args.old_lib)

    new = torch.ops._qwen38_C.moe_compact_down_grouped_gemm
    old = getattr(torch.ops.moe_int4_ops,
                  f"moe_compact{args.k}_down_grouped_gemm")
    x = torch.randn((args.m, args.k), dtype=torch.float16, device="xpu")
    weight = torch.randint(0, 256, (512, 2560, args.k // 2),
                           dtype=torch.uint8, device="xpu").view(torch.int8)
    scale = torch.rand((512, 2560, (args.k + 127) // 128),
                       dtype=torch.float16, device="xpu") * 0.1
    counts = torch.zeros(512, dtype=torch.int32, device="xpu")
    # Test both sparse expert occupancy and non-multiple-of-eight tails.
    active = min(64, args.m)
    counts[:active] = args.m // active
    counts[:args.m % active] += 1

    expected = old(x, weight, scale, counts)
    actual = new(x, weight, scale, counts)
    torch.xpu.synchronize()
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.01)
    print(f"K={args.k} M={args.m} old_ms={measure(old, x, weight, scale, counts, args.repeats):.4f} "
          f"new_ms={measure(new, x, weight, scale, counts, args.repeats):.4f}")


if __name__ == "__main__":
    main()
