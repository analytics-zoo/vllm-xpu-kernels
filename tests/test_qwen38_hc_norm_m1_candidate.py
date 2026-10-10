"""M1 combine-norm FP16 boundary: old/new DSOs run in separate processes.

Set QWEN38_HC_OLD_DSO and QWEN38_HC_CANDIDATE_DSO, plus one-card affinity,
then run this file with pytest after GPU/build authorization. No two native
DSOs sharing qwen38_hc_sycl are loaded into one process.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

OLD_DSO_SHA256 = (
    "540868ec515b56e2e6a360b1ba5c72544cb656c36c3ff4710242fceca15e6dff"
)


def _file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _make_midpoint_case(torch, seed):
    # Injection zero gives scale exactly one. Select half inputs whose FP32
    # sums are 0.35--0.43 half-ULP from the nearest half: enough signal for
    # the FP16 boundary, but at least 0.07 ULP from a rounding midpoint.
    torch.manual_seed(seed)
    first = (torch.randn(65536, dtype=torch.float32) * 0.2).half()
    second = (torch.randn(65536, dtype=torch.float32) * 0.2).half()
    full = first.float() + second.float()
    rounded = full.half().float()
    magnitude = rounded.abs()
    ulp = torch.pow(2.0, torch.floor(torch.log2(magnitude)) - 10)
    distance = (full - rounded).abs() / ulp
    selected = ((distance >= 0.35) & (distance <= 0.43) &
                (magnitude >= 0.02) & (magnitude <= 0.8)).nonzero().flatten()
    if selected.numel() < 2560:
        raise AssertionError(f"insufficient midpoint-margin pairs: {seed}")
    indices = selected[:2560]
    hidden_cpu = first[indices].repeat(4).view(1, 10240)
    block_cpu = second[indices].view(1, 2560)
    raw = hidden_cpu.float().view(1, 4, 2560) + block_cpu.float()[:, None, :]
    combined_cpu = raw.half().reshape(1, 10240)
    return hidden_cpu, block_cpu, raw, combined_cpu


def _norm_golden(torch, values):
    grouped = values.reshape(1, 4, 2560)
    inv = torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-6)
    return (grouped * inv).reshape(1, 10240).half()


def _tensor_hash(tensor):
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


def _worker(library):
    import torch

    if not torch.xpu.is_available():
        raise RuntimeError("XPU unavailable")
    torch.xpu.set_device("xpu:0")
    torch.ops.load_library(str(Path(library).resolve(strict=True)))
    op = torch.ops.qwen38_hc_sycl.combine_norm
    results = []
    for seed in (7101, 7102, 7103):
        hidden_cpu, block_cpu, raw, combined_cpu = _make_midpoint_case(
            torch, seed)
        hidden = hidden_cpu.to("xpu")
        block = block_cpu.to("xpu")
        injection = torch.zeros((1, 4), device="xpu", dtype=torch.float16)
        weight = torch.zeros((10240, ), device="xpu", dtype=torch.float16)
        combined = torch.empty_like(hidden)
        normed = torch.empty_like(hidden)
        op(hidden, block, injection, weight, combined, normed, 1e-6)
        torch.xpu.synchronize()
        combined_host = combined.cpu()
        normed_host = normed.cpu()
        if not torch.equal(combined_host, combined_cpu):
            raise AssertionError(
                f"combined half bits differ from golden: {seed}"
            )
        rounded = _norm_golden(torch, combined_cpu.float())
        unrounded = _norm_golden(torch, raw)
        rounded_mismatch = int((normed_host != rounded).sum())
        unrounded_mismatch = int((normed_host != unrounded).sum())
        if rounded_mismatch > 16 or unrounded_mismatch < 500:
            raise AssertionError(
                f"FP16 norm boundary failed seed={seed}: "
                f"rounded={rounded_mismatch}, unrounded={unrounded_mismatch}"
            )
        results.append({
            "seed": seed,
            "combined_sha256": _tensor_hash(combined_host),
            "normed_sha256": _tensor_hash(normed_host),
            "rounded_mismatch": rounded_mismatch,
            "unrounded_mismatch": unrounded_mismatch,
        })
    print(json.dumps({"kind": "norm_m1_signature", "results": results}))


def _signature(library):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               str(library)]
    completed = subprocess.run(command, capture_output=True, text=True,
                               check=False)
    if completed.returncode:
        raise AssertionError(
            f"isolated DSO worker failed: {library}\n{completed.stderr}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])["results"]


def test_old_new_exact_combined_and_normed_with_midpoint_discriminator():
    old = os.environ.get("QWEN38_HC_OLD_DSO")
    candidate = os.environ.get("QWEN38_HC_CANDIDATE_DSO")
    if not old or not candidate:
        pytest.skip("set distinct old/candidate HC DSO paths")
    if os.environ.get("ZE_AFFINITY_MASK") not in {"4", "5", "6", "7"}:
        pytest.skip("requires one authorized physical GPU via ZE_AFFINITY_MASK")
    assert (Path(old).resolve(strict=True) !=
            Path(candidate).resolve(strict=True))
    assert _file_hash(old) == OLD_DSO_SHA256
    assert _file_hash(candidate) != OLD_DSO_SHA256
    assert _signature(old) == _signature(candidate)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit(
            "usage: test_qwen38_hc_norm_m1_candidate.py --worker DSO"
        )
    _worker(sys.argv[2])
