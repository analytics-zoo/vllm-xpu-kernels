"""Small-M manual Norm: CPU checks plus isolated, strictly compared DSOs."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys

import pytest

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "hc_norm_m1_math_helpers",
    Path(__file__).with_name("test_qwen38_hc_norm_m1_candidate.py"),
)
assert _HELPER_SPEC is not None and _HELPER_SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPERS)
_file_hash = _HELPERS._file_hash
_make_midpoint_case = _HELPERS._make_midpoint_case
_tensor_hash = _HELPERS._tensor_hash

BASE544_SHA256 = (
    "54437719e441d4db53d36b3dc4357db63094784825590ea3dd5c0d2012919116"
)
ROWS = (2, 3, 4, 5, 8)
STRIDES = (4, 17, 336)


@pytest.mark.parametrize("m", range(2, 9))
def test_cpu_group_row_branch_and_stride_coverage(m):
    for stride in STRIDES:
        outputs = set()
        injections = set()
        for group in range(m * 4):
            row, branch = divmod(group, 4)
            assert group * 2560 == row * 10240 + branch * 2560
            outputs.update(range(group * 2560, (group + 1) * 2560))
            injections.add(row * stride + branch)
            assert 0 <= row * 2560 + 2559 < m * 2560
            assert 0 <= branch * 2560 + 2559 < 10240
        assert len(outputs) == m * 10240
        assert len(injections) == m * 4


def test_cpu_manual_normal_rne_bits_against_independent_half():
    generator = random.Random(380544)
    words = [generator.randrange(0x38800000, 0x477FF000)
             for _ in range(10000)]
    # Exact half midpoints, adjacent FP32 values, both signs and parities.
    for half_bits in (0x0400, 0x0401, 0x3C00, 0x3C01, 0x7BFD):
        low = struct.unpack("<e", struct.pack("<H", half_bits))[0]
        high = struct.unpack("<e", struct.pack("<H", half_bits + 1))[0]
        midpoint = struct.unpack("<I", struct.pack("<f", (low + high) / 2))[0]
        words.extend((midpoint - 1, midpoint, midpoint + 1))
    for magnitude in words:
        for sign in (0, 0x80000000):
            bits = magnitude | sign
            value = struct.unpack("<f", struct.pack("<I", bits))[0]
            expected = struct.unpack("<H", struct.pack("<e", value))[0]
            rounded = magnitude + 0xFFF + ((magnitude >> 13) & 1)
            actual = ((bits >> 16) & 0x8000) | (
                (rounded - 0x38000000) >> 13)
            feedback_bits = sign | (rounded & 0xFFFFE000)
            feedback = struct.unpack("<f", struct.pack("<I", feedback_bits))[0]
            assert actual == expected
            expected_float = struct.unpack(
                "<e", struct.pack("<H", expected))[0]
            assert feedback == expected_float


def _golden(torch, combined, weight):
    m = combined.shape[0]
    values = combined.float().reshape(m, 4, 2560)
    inv = torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    return ((values * inv) *
            (1 + weight.float().reshape(1, 4, 2560))).reshape(m, 10240).half()


def _cpu_case(torch, m, seed, kind):
    torch.manual_seed(seed)
    if kind == "midpoint":
        rows = [_make_midpoint_case(torch, seed + row) for row in range(m)]
        hidden = torch.cat([case[0] for case in rows])
        block = torch.cat([case[1] for case in rows])
        raw = torch.cat([case[2] for case in rows])
        injection = torch.zeros((m, 4), dtype=torch.float16)
        weight = torch.zeros(10240, dtype=torch.float16)
    elif kind == "ties":
        # Exactly representable FP32 sums at even/odd half midpoints.
        hidden = torch.empty((m, 4, 2560), dtype=torch.float16)
        block = torch.empty((m, 2560), dtype=torch.float16)
        for row in range(m):
            sign = -1 if row % 2 else 1
            block[row].fill_(sign * 2 ** -11)
            for branch in range(4):
                hidden[row, branch].fill_(sign * (1 + (branch % 2) * 2 ** -10))
        hidden = hidden.reshape(m, 10240)
        injection = torch.zeros((m, 4), dtype=torch.float16)
        weight = torch.zeros(10240, dtype=torch.float16)
        raw = hidden.float().reshape(m, 4, 2560) + block.float()[:, None, :]
    else:
        hidden = (torch.randn((m, 10240)) * 0.2).half()
        block = (torch.randn((m, 2560)) * 0.2).half()
        injection = torch.randn((m, 4)).half()
        weight = (torch.randn(10240) * 0.03).half()
        scale = 2 * (1 / (1 + torch.exp(-injection.float() * 0.25)))
        raw = hidden.float().reshape(m, 4, 2560) + (
            block.float()[:, None, :] * scale[:, :, None])
        rounded = raw.half().float()
        halves = rounded.half()
        lower = torch.nextafter(
            halves, torch.full_like(halves, -float("inf"))).float()
        upper = torch.nextafter(
            halves, torch.full_like(halves, float("inf"))).float()
        margin = torch.minimum(
            (raw - (lower + rounded) * 0.5).abs(),
            (raw - (upper + rounded) * 0.5).abs())
        # Remove only numerically ambiguous CPU-vs-SYCL exp midpoint columns.
        # No tolerance relaxation: every combined half must remain exact.
        unsafe = (margin < 2e-6).any(dim=1)
        if kind != "random_full":
            block.masked_fill_(unsafe, 0)
        raw = hidden.float().reshape(m, 4, 2560) + (
            block.float()[:, None, :] * scale[:, :, None])
    return hidden, block, injection, weight, raw, raw.half().reshape(m, 10240)


@pytest.mark.parametrize("m", ROWS)
def test_cpu_strong_discriminator_and_random_case_generation(m):
    import torch

    for kind in ("random", "random_full", "midpoint", "ties"):
        h, b, inj, w, raw, expected = _cpu_case(torch, m, 7101, kind)
        assert all(t.device.type == "cpu" for t in (h, b, inj, w))
        assert expected.shape == (m, 10240)
        assert torch.equal(expected, raw.half().reshape(m, 10240))
        if kind == "midpoint":
            rounded = _golden(torch, expected, w)
            unrounded = _golden(torch, raw.reshape(m, 10240), w)
            assert int((rounded != unrounded).sum()) >= 500 * m


def _worker(library, m):
    import torch

    if os.environ.get("ZE_AFFINITY_MASK") != "7":
        raise RuntimeError("requires separately authorized GPU7")
    torch.xpu.set_device("xpu:0")
    torch.ops.load_library(str(Path(library).resolve(strict=True)))
    op = torch.ops.qwen38_hc_sycl.combine_norm
    results = []
    for seed in (7101, 7102, 7103):
        for kind in ("random", "random_full", "midpoint", "ties"):
            h, b, inj, w, raw, expected = _cpu_case(torch, m, seed, kind)
            for stride in STRIDES:
                hidden, block, weight = h.to("xpu"), b.to("xpu"), w.to("xpu")
                storage = torch.full((m, stride), -7, device="xpu",
                                     dtype=torch.float16)
                injection = storage[:, :4]
                injection.copy_(inj)
                assert injection.stride() == (stride, 1)
                combined = torch.empty_like(hidden)
                normed = torch.empty_like(hidden)
                op(hidden, block, injection, weight, combined, normed, 1e-6)
                torch.xpu.synchronize()
                actual_combined, actual_normed = combined.cpu(), normed.cpu()
                if kind != "random_full":
                    assert torch.equal(actual_combined.view(torch.int16),
                                       expected.view(torch.int16)), (
                                           m, seed, kind, stride)
                rounded = _golden(torch, expected, w)
                mismatch = int((actual_normed != rounded).sum())
                if kind != "random_full":
                    assert mismatch <= 16 * m, (
                        m, seed, kind, stride, mismatch)
                if kind == "midpoint":
                    unrounded = _golden(torch, raw.reshape(m, 10240), w)
                    # _golden converts its input to FP32, without half-rounding.
                    bad = int((actual_normed != unrounded).sum())
                    assert bad >= 500 * m
                    assert bad >= 10 * max(mismatch, 1)
                results.append({
                    "m": m, "seed": seed, "kind": kind, "stride": stride,
                    "input_sha256": hashlib.sha256(
                        b"".join(t.numpy().tobytes() for t in (h, b, inj, w))
                    ).hexdigest(),
                    "combined_sha256": _tensor_hash(actual_combined),
                    "normed_sha256": _tensor_hash(actual_normed),
                    "rounded_mismatch": mismatch,
                    "independent_combined_bits_exact": kind != "random_full",
                })
    print(json.dumps({"results": results,
                      "library_sha256": _file_hash(library)}))


def _signature(library, m):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               str(library), str(m)]
    completed = subprocess.run(command, capture_output=True, text=True,
                               check=False)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("m", ROWS)
def test_old544_new_exact_combined_normed_and_strong_golden(m):
    old = os.environ.get("QWEN38_HC_OLD_DSO")
    candidate = os.environ.get("QWEN38_HC_CANDIDATE_DSO")
    if not old or not candidate or os.environ.get("ZE_AFFINITY_MASK") != "7":
        pytest.skip("requires old544/candidate DSOs and authorized GPU7")
    assert (Path(old).resolve(strict=True) !=
            Path(candidate).resolve(strict=True))
    assert _file_hash(old) == BASE544_SHA256
    assert _file_hash(candidate) != BASE544_SHA256
    a, b = _signature(old, m), _signature(candidate, m)
    assert a["library_sha256"] == BASE544_SHA256
    assert b["library_sha256"] == _file_hash(candidate)
    # Includes original, UNFILTERED random columns: both full-output hashes
    # must match. Any normed/combined bit mismatch rejects the candidate.
    assert a["results"] == b["results"]


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit(
            "usage: test_qwen38_hc_norm_smallm_candidate.py --worker DSO M")
    _worker(sys.argv[2], int(sys.argv[3]))
