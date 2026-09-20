# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

import json
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)

from tests.ops.layernorm_op import RMSNorm
from tests.utils import STR_DTYPE_TO_TORCH_DTYPE


@torch.inference_mode()
def benchmark_gemma_decode(hidden_size, heads, add_residual, dtype, num_iters,
                           num_warmup_iters, result_json):
    """Compare the native chain and XPU path, including weight preparation."""
    import vllm.kernels  # noqa: F401
    from vllm import ir
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    if result_json and Path(result_json).exists():
        raise FileExistsError(result_json)
    shape = (1, heads, hidden_size) if heads > 1 else (1, hidden_size)
    if add_residual and heads != 1:
        raise ValueError("Residual decode cases use [1, hidden_size]")
    torch.manual_seed(0)
    layer = GemmaRMSNorm(hidden_size).to(device="xpu", dtype=dtype)
    layer.weight.normal_(mean=0.0, std=0.1)
    op = ir.ops.fused_add_rms_norm if add_residual else ir.ops.rms_norm

    def pool(count, seed):
        torch.manual_seed(seed)
        x = torch.randn(shape, device="xpu", dtype=dtype)
        residual = torch.randn_like(x) if add_residual else None
        # Each iteration gets fresh values: in-place residual updates must not
        # accumulate across iterations. Preparation stays outside timing.
        return [(x.clone(), residual.clone() if residual is not None else None)
                for _ in range(count)]

    def measure(provider, seed, count):
        inputs = pool(count, seed)
        fn = layer.forward_native if provider == "native" else layer.forward_xpu
        with op.set_priority([provider]):
            torch.xpu.synchronize()
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
            begin = time.perf_counter()
            for x, residual in inputs:
                fn(x, residual)
            end.record()
            torch.xpu.synchronize()
            wall_us = (time.perf_counter() - begin) * 1e6 / count
            device_us = start.elapsed_time(end) * 1000 / count
        return {"wall_us": wall_us, "device_interval_us": device_us}

    results = {}
    for provider in ("native", "xpu_kernels"):
        measure(provider, 41, num_warmup_iters)
        results[provider] = [
            measure(provider, seed, num_iters) for seed in (42, 43, 44)
        ]
    baseline = sum(r["wall_us"] for r in results["native"]) / 3
    candidate = sum(r["wall_us"] for r in results["xpu_kernels"]) / 3
    report = {
        "shape":
        shape,
        "dtype":
        str(dtype),
        "add_residual":
        add_residual,
        "iterations_per_sample":
        num_iters,
        "seeds": [42, 43, 44],
        "samples":
        results,
        "native_mean_us":
        baseline,
        "xpu_mean_us":
        candidate,
        "reduction_pct": (1 - candidate / baseline) * 100,
        "measurement":
        "Resident full Norm call including float weight + 1; "
        "block-boundary synchronization, no per-op synchronization; "
        "XPU event interval includes stream idle time between launches.",
    }
    if result_json:
        Path(result_json).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


@torch.inference_mode()
def main(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int = 0,
    num_warmup_iters: int = 5,
    num_iters: int = 100,
) -> None:
    torch.set_default_device("xpu")

    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None

    def run_xpu_benchmark(num_iters: int) -> float:
        torch.xpu.synchronize()

        start_time = time.perf_counter()

        for _ in range(num_iters):
            layer(x, residual)
        torch.xpu.synchronize()

        end_time = time.perf_counter()

        return (end_time - start_time) / num_iters

    # Warmup.
    print("Warming up...")
    run_benchmark = run_xpu_benchmark
    run_benchmark(num_iters=num_warmup_iters)

    # Benchmark.
    latency = run_benchmark(num_iters=num_iters)
    print(f"Kernel running time: {latency * 1000000:.3f} us")


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the layernorm kernel.")
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--add-residual", action="store_true")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16", "float"],
                        default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-warmup-iters", type=int, default=5)
    parser.add_argument("--num-iters",
                        type=int,
                        default=100,
                        help="Number of benchmark iterations. ")
    parser.add_argument(
        "--gemma-decode", action="store_true",
        help="Compare native and XPU Gemma Norm calls; requires vLLM.")
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--result-json", type=str)

    args = parser.parse_args()
    print(args)

    if args.gemma_decode:
        from vllm.config import VllmConfig, set_current_vllm_config

        with set_current_vllm_config(VllmConfig()):
            benchmark_gemma_decode(args.hidden_size, args.heads,
                                   args.add_residual,
                                   STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
                                   args.num_iters, args.num_warmup_iters,
                                   args.result_json)
    else:
        main(
            num_tokens=args.num_tokens,
            hidden_size=args.hidden_size,
            add_residual=args.add_residual,
            dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
            seed=args.seed,
            num_warmup_iters=args.num_warmup_iters,
            num_iters=args.num_iters,
        )
