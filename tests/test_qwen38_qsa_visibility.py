# SPDX-License-Identifier: Apache-2.0
"""SelectionPhase0 invisible-partition regression; GPU comparison is opt-in.

Run CPU geometry/math now with pytest. After GPU6 is granted, set
QWEN38_VISIBILITY_GPU6_APPROVED=1, ZE_AFFINITY_MASK=6, and the OLD/NEW
library paths below. Each DSO is loaded in a separate process to avoid
duplicate Torch-op registration. Full output bytes must have equal SHA256;
all tie-case rows also meet an independent exact binary-score/tie reference.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROWS = (128, 129, 4096)
LENGTHS = (4096, 65536, 65537, 128000, 256000)
PAGE_SIZES = (64, 128)  # TP8/TP4 compressed-key page geometries.
BOUNDARIES = (-3, -1, 0, 1, 2, 3, 4, 7, 2047, 2048, 2050, 2051,
              4095, 4096, 65535, 65536, 65537)


def geometry(rows, max_seq_len):
    partitions = 32 if max_seq_len >= 4096 else 1
    max_blocks = (max_seq_len + 3) // 4
    per_partition = (max_blocks + partitions - 1) // partitions
    blocks_per_partition = ((per_partition + 511) // 512) * 512
    return partitions, blocks_per_partition, (rows + 127) // 128


def positions_for(rows, length, window):
    if window == "boundary":
        anchors = BOUNDARIES + (length // 2, length - 2, length - 1,
                                length, length + 3)
        return [anchors[row % len(anchors)] for row in range(rows)]
    starts = {"early": 0, "middle": (length - rows) // 2,
              "tail": length - rows}
    start = starts[window]
    return list(range(start, start + rows))


def empty_workgroups(positions, length, sequence_lengths=None, requests=None):
    partitions, blocks_per_partition, _ = geometry(len(positions), length)
    sequences = sequence_lengths or [length]
    reqs = requests or [0] * len(positions)
    empty = 0
    for pos, req in zip(positions, reqs):
        seq = max(sequences[req], 0) if 0 <= req < len(sequences) else 0
        visible_blocks = min(max(pos + 1, 0), seq) // 4
        empty += sum(partition * blocks_per_partition >= visible_blocks
                     for partition in range(partitions))
    return empty, len(positions) * partitions


def exact_binary_row(position, sequence_length):
    """Independent integer/FP16-exact score reference; no BLAS or topk."""
    result = [-1] * 2051
    visible = max(position + 1, 0)
    seq = max(sequence_length, 0)
    blocks = min(visible, seq) // 4
    high = [block for block in range(blocks) if block % 7 == 0]
    low = [block for block in range(blocks) if block % 7 != 0]
    selected = (high + low)[:512]
    for rank, block in enumerate(selected):
        result[rank * 4:rank * 4 + 4] = [block * 4 + offset
                                         for offset in range(4)]
    tail_start = visible // 4 * 4
    for offset in range(visible - tail_start):
        token = tail_start + offset
        if token < seq:
            result[len(selected) * 4 + offset] = token
    return result


def exact_binary_batch(positions, sequence_lengths, requests):
    """Vectorized independent score ordering and tail math for every row."""
    import numpy as np

    positions = np.asarray(positions, dtype=np.int64)
    sequences = np.asarray(sequence_lengths, dtype=np.int64)
    requests = np.asarray(requests, dtype=np.int64)
    seq = np.zeros(len(positions), dtype=np.int64)
    valid_request = (requests >= 0) & (requests < len(sequences))
    seq[valid_request] = np.maximum(sequences[requests[valid_request]], 0)
    visible = np.maximum(positions + 1, 0)
    blocks = np.minimum(visible, seq) // 4
    selected = np.minimum(blocks, 512)
    high_count = np.minimum((blocks + 6) // 7, 512)
    ranks = np.arange(512, dtype=np.int64)
    # The binary fixture scores multiples of seven above all other blocks;
    # negative dot products are ReLU-clipped to the zero-score tie group.
    low_blocks = np.flatnonzero(np.arange(600) % 7 != 0)[:512]
    low_rank = np.clip(ranks[None, :] - high_count[:, None], 0, 511)
    ordered = np.where(ranks[None, :] < high_count[:, None],
                       7 * ranks[None, :], low_blocks[low_rank])
    ordered = np.where(ranks[None, :] < selected[:, None], ordered, -1)
    tokens = ordered[:, :, None] * 4 + np.arange(4)
    tokens = np.where(ordered[:, :, None] >= 0, tokens, -1)
    expected = np.full((len(positions), 2051), -1, dtype=np.int32)
    expected[:, :2048] = tokens.reshape(len(positions), 2048)
    tail_start = visible // 4 * 4
    tail_count = visible - tail_start
    for offset in range(3):
        token = tail_start + offset
        valid = (offset < tail_count) & (token < seq)
        slot = np.where(selected < 512, selected * 4 + offset,
                        2048 + offset)
        row_ids = np.flatnonzero(valid)
        expected[row_ids, slot[row_ids]] = token[row_ids]
    return expected


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("length", LENGTHS)
def test_cpu_geometry_and_visibility(rows, length):
    partitions, block, chunks = geometry(rows, length)
    assert partitions == 32
    assert chunks == (rows + 127) // 128
    assert block == (512 if length <= 65536 else 1024 if length <= 131072
                     else 2048)
    for window in ("early", "middle", "tail", "boundary"):
        positions = positions_for(rows, length, window)
        empty, total = empty_workgroups(positions, length)
        assert 0 <= empty <= total == rows * 32
    if length > rows:
        early = empty_workgroups(positions_for(rows, length, "early"),
                                 length)[0]
        late = empty_workgroups(positions_for(rows, length, "tail"),
                                length)[0]
        assert early >= late


def test_cpu_single_partition_and_exact_boundary_math():
    assert geometry(1, 1024)[0] == 1  # Fast path must not touch direct tail.
    assert geometry(128, 65536)[1] == 512
    assert geometry(128, 65537)[1] == 1024
    assert exact_binary_row(-3, 4096) == [-1] * 2051
    assert exact_binary_row(-1, 4096) == [-1] * 2051
    assert exact_binary_row(2, 4096)[:5] == [0, 1, 2, -1, -1]
    assert exact_binary_row(3, 4096)[:5] == [0, 1, 2, 3, -1]
    assert exact_binary_row(27, 4096)[:8] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert exact_binary_row(4096, 4096)[2048:] == [-1, -1, -1]
    for pos in (2046, 2047, 2048, 2050, 2051, 65535, 65536, 65537):
        empty, total = empty_workgroups([pos], 65537)
        assert 0 <= empty <= total == 32
    assert empty_workgroups([-3, -1, 0], 65537)[0] == 96
    assert empty_workgroups([0], 65537, [-1])[0] == 32
    assert empty_workgroups([0], 65537, [65537], [-1])[0] == 32
    positions = BOUNDARIES + (128000, 256000)
    batched = exact_binary_batch(positions, [256000], [0] * len(positions))
    for row, position in enumerate(positions):
        assert batched[row].tolist() == exact_binary_row(position, 256000)


def cases():
    for rows in ROWS:
        for length in LENGTHS:
            for page_size in PAGE_SIZES:
                for window in ("early", "middle", "tail"):
                    yield rows, length, page_size, window, "tie"
    for rows in ROWS:
        for page_size in PAGE_SIZES:
            yield rows, 65537, page_size, "boundary", "tie"
    # The auxiliary fused QNorm/RoPE/select entry reaches the same kernel.
    for rows in (128, 129):
        for page_size in PAGE_SIZES:
            for window in ("early", "tail"):
                yield rows, 65537, page_size, window, "fused_tie"
    for window in ("early", "tail"):
        yield 4096, 128000, 128, window, "fused_tie"
    # Non-tie scores still require old/new full-output byte equality.
    for window in ("early", "middle", "tail"):
        yield 4096, 128000, 128, window, "random"


def gpu_worker(library, output_dir):
    import numpy as np
    import torch

    spec = importlib.util.spec_from_file_location("_qwen38_C", library)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if getattr(module, "qsa_sycl_selection_wide_scratch_abi_version", 0) != 1:
        raise RuntimeError("expected wide scratch ABI=1")
    device = "xpu:0"
    records = []
    for case_index, case in enumerate(cases()):
        rows, length, page_size, window, kind = case
        torch.manual_seed(9001 + rows + length + page_size)
        pages = (length + 4 * page_size - 1) // (4 * page_size)
        if kind in ("tie", "fused_tie"):
            query = torch.zeros(rows, 4, 128, dtype=torch.float16)
            query[:, 0, 0] = 1
            cache = torch.zeros(pages, page_size, 1, 128,
                                dtype=torch.float16)
            ids = torch.arange(pages * page_size).reshape(pages, page_size)
            cache[:, :, 0, 0] = ((ids % 7 == 0).half() -
                                 ((ids % 7 != 0) & (ids % 11 == 0)).half())
        else:
            query = torch.randn(rows, 4, 128, dtype=torch.float16)
            cache = torch.randn(pages, page_size, 1, 128,
                                dtype=torch.float16)
        positions = positions_for(rows, length, window)
        requests = [0] * rows
        sequences = [length]
        if window == "boundary":
            sequences = [length, min(length, 4097), -1]
            requests = [row % 4 if row % 4 < 3 else -1
                        for row in range(rows)]
        table = torch.arange(pages, dtype=torch.int32).repeat(
            len(sequences), 1)
        shape = (128, 32, 512)
        workspace = tuple(torch.empty(shape, dtype=dtype, device=device)
                          for dtype in (torch.float32, torch.int32,
                                        torch.float32, torch.int32))
        output = torch.empty(rows, 2051, dtype=torch.int32, device=device)
        xpu_query = query.to(device)
        xpu_cache = cache.to(device)
        xpu_table = table.to(device)
        xpu_requests = torch.tensor(requests, dtype=torch.int32,
                                    device=device)
        xpu_positions = torch.tensor(positions, dtype=torch.int64,
                                     device=device)
        xpu_sequences = torch.tensor(sequences, dtype=torch.int32,
                                     device=device)
        if kind == "fused_tie":
            weight = torch.zeros(128, dtype=torch.float16, device=device)
            rope_positions = torch.zeros(rows, dtype=torch.int32,
                                         device=device)
            rope = torch.zeros(1, 64, dtype=torch.float16, device=device)
            rope[:, :32] = 1
            q_output = torch.empty_like(xpu_query)
            module.qsa_sycl_q_norm_rope_select_v1(
                xpu_query, weight, rope_positions, rope, xpu_cache,
                xpu_table, xpu_requests, xpu_positions, xpu_sequences,
                q_output, output, *workspace, page_size, length,
                False, False, True)
        else:
            module.select_paged_tokens_v2(
                xpu_query, xpu_cache, xpu_table, xpu_requests,
                xpu_positions, xpu_sequences, page_size, length,
                output, *workspace)
        actual = output.cpu()
        np.savez_compressed(output_dir / f"{case_index:03}.npz",
                            output=actual.numpy())
        if kind in ("tie", "fused_tie"):
            expected = exact_binary_batch(positions, sequences, requests)
            assert np.array_equal(actual.numpy(), expected), (
                rows, length, page_size, window)
        output_sha = hashlib.sha256(actual.numpy().tobytes()).hexdigest()
        records.append({
            "case": [rows, length, page_size, window, kind],
            "output_sha256": output_sha,
            "independent_math_all_rows": kind in ("tie", "fused_tie"),
            "empty_workgroups": empty_workgroups(
                positions, length, sequences, requests),
        })
    library_sha = hashlib.sha256(Path(library).read_bytes()).hexdigest()
    print("VISIBILITY_JSON=" + json.dumps({
        "library_sha256": library_sha,
        "records": records,
    }), flush=True)


def test_gpu_old_new_indices_exact_and_independent_math(tmp_path):
    if os.environ.get("QWEN38_VISIBILITY_GPU6_APPROVED") != "1":
        pytest.skip("GPU6 not yet granted for visibility candidate")
    assert os.environ.get("ZE_AFFINITY_MASK") == "6"
    old = os.environ["QWEN38_QSA_VISIBILITY_OLD_LIBRARY"]
    new = os.environ["QWEN38_QSA_VISIBILITY_NEW_LIBRARY"]
    outputs = []
    for label, library in (("old", old), ("new", new)):
        assert Path(library).is_file()
        output_dir = tmp_path / label
        output_dir.mkdir()
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker",
             library, str(output_dir)], check=False, text=True,
            capture_output=True)
        assert completed.returncode == 0, (
            f"{label} worker failed:\n{completed.stdout}\n"
            f"{completed.stderr}")
        line = next(line.removeprefix("VISIBILITY_JSON=")
                    for line in completed.stdout.splitlines()
                    if line.startswith("VISIBILITY_JSON="))
        outputs.append(json.loads(line))
    assert outputs[0]["library_sha256"] != outputs[1]["library_sha256"]
    assert outputs[0]["records"] == outputs[1]["records"]
    import numpy as np

    case_results = []
    for index in range(len(outputs[0]["records"])):
        with (np.load(tmp_path / "old" / f"{index:03}.npz") as old_output,
              np.load(tmp_path / "new" / f"{index:03}.npz") as new_output):
            equal = np.array_equal(old_output["output"],
                                   new_output["output"])
            case_results.append({"case_index": index,
                                 **outputs[0]["records"][index],
                                 "indices_exact_equal": bool(equal)})
            assert equal, index
    result_path = os.environ.get("QWEN38_QSA_VISIBILITY_RESULT")
    if result_path:
        with Path(result_path).open("x", encoding="utf-8") as handle:
            json.dump({"old_library_sha256": outputs[0]["library_sha256"],
                       "new_library_sha256": outputs[1]["library_sha256"],
                       "case_count": len(case_results),
                       "cases": case_results}, handle, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit(
            "usage: test_qwen38_qsa_visibility.py --worker DSO OUT")
    if (os.environ.get("QWEN38_VISIBILITY_GPU6_APPROVED") != "1" or
            os.environ.get("ZE_AFFINITY_MASK") != "6"):
        raise SystemExit("GPU6 visibility candidate is not approved")
    gpu_worker(sys.argv[2], Path(sys.argv[3]))
