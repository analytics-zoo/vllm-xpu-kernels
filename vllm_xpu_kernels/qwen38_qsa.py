# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 QSA portable-SYCL provider.

Only actual semantic capabilities are advertised. Scratch is owned by this
provider, keyed by XPU stream; C++ owns neither allocations nor fallback.
"""

from __future__ import annotations

import threading

import torch

from . import _qwen38_C

qsa_sycl_abi_version = 1
activation_dtype = "float16"
selection_page_sizes = (64, 128)
attention_page_sizes = (256, 512)
selection_output_width = 2051
packed_kv = 1
# Read the native attribute once at import. An old DSO has no marker and
# therefore keeps legacy 32-row scratch; no per-token Pybind call is added.
qsa_sycl_selection_wide_scratch_abi_version = int(
    getattr(_qwen38_C, "qsa_sycl_selection_wide_scratch_abi_version", 0)
)

# These describe semantic coverage, not an ESIMD tile/subgroup algorithm.
qsa_sycl_group_compress_v2 = 1
qsa_sycl_token_split_v3 = 1
qsa_sycl_selection_host_max_seq_len = 1
qsa_sycl_selection_max_rows = 4096
qsa_sycl_attention_max_rows = 4096
qsa_sycl_q6 = 1
qsa_token_split_candidate_page512 = 1
qsa_sycl_qkv_postprocess_v1 = 1
qsa_sycl_indexer_norm_rope_v1 = 1
qsa_sycl_indexer_norm_rope_v2 = 1
qsa_sycl_indexer_projection_int4_max_rows = 8
qsa_sycl_q_norm_rope_select_v1 = 1
qsa_sycl_row_store_v3 = int(
    callable(getattr(_qwen38_C, "qsa_sycl_store_cache_rows_v3", None))
)
qsa_sycl_row_store_parallel_v1 = qsa_sycl_row_store_v3
qsa_store_m1_transaction_abi_version = int(
    callable(getattr(_qwen38_C, "qsa_sycl_try_store_m1_transaction_v1", None))
)
_m1_fused_store = getattr(
    _qwen38_C, "qsa_sycl_try_store_m1_transaction_fused_v1", None)
if not callable(_m1_fused_store):
    _m1_fused_store = None

# The existing vLLM TP4 q6 gate checks this semantic ABI. The same core
# implements 6 query heads, one packed KV head, page 256/512 and v3 partials.
qsa_q6_abi_version = 1
qsa_q6_query_heads = 6
qsa_q6_kv_heads = 1
qsa_q6_token_split_v3 = 1
qsa_q6_page512 = 1


class _StreamScratch:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._selection: dict[
            tuple[int, int, int], tuple[torch.Tensor, ...]
        ] = {}
        self._attention: dict[tuple[int, int, int, int], torch.Tensor] = {}
        self._normalized_q: dict[tuple[int, int, int], torch.Tensor] = {}

    @staticmethod
    def _stream_key(tensor: torch.Tensor) -> tuple[int, int]:
        stream = torch.xpu.current_stream(tensor.device)
        return int(stream.device_index), int(stream.stream_id)

    def selection(self, q: torch.Tensor,
                  max_seq_len: int) -> tuple[torch.Tensor, ...]:
        rows = int(q.shape[0])
        scratch_rows = min(rows, 32)
        if (rows >= 128 and max_seq_len >= 4096
                and qsa_sycl_selection_wide_scratch_abi_version == 1):
            scratch_rows = 128
        key = (*self._stream_key(q), scratch_rows)
        with self._lock:
            scratch = self._selection.get(key)
            if scratch is None:
                shape = (key[-1], 32, 512)
                scratch = (
                    torch.empty(shape, dtype=torch.float32, device=q.device),
                    torch.empty(shape, dtype=torch.int32, device=q.device),
                    torch.empty(shape, dtype=torch.float32, device=q.device),
                    torch.empty(shape, dtype=torch.int32, device=q.device),
                )
                self._selection[key] = scratch
            return scratch

    def attention(self, q: torch.Tensor) -> torch.Tensor:
        key = (*self._stream_key(q), min(int(q.shape[0]), 128), int(q.shape[1]))
        with self._lock:
            scratch = self._attention.get(key)
            if scratch is None:
                scratch = torch.empty(
                    (key[2], key[3], 43, 258),
                    dtype=torch.float32,
                    device=q.device,
                )
                self._attention[key] = scratch
            return scratch

    def normalized_q(self, q: torch.Tensor) -> torch.Tensor:
        key = (*self._stream_key(q), int(q.shape[0]))
        with self._lock:
            scratch = self._normalized_q.get(key)
            if scratch is None:
                scratch = torch.empty(
                    (key[2], 4, 128), dtype=torch.float16, device=q.device
                )
                self._normalized_q[key] = scratch
            return scratch


_scratch = _StreamScratch()


def qsa_select_paged_tokens_v2(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    compressed_page_size: int,
    out: torch.Tensor,
    *,
    max_seq_len: int,
) -> torch.Tensor:
    """Old positional ABI plus mandatory trusted-host sequence bound."""
    if token_topk != 2048 or compress_ratio != 4:
        raise ValueError("SYCL QSA selection requires topk=2048, ratio=4")
    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError("SYCL QSA selection requires host max_seq_len")
    return _qwen38_C.select_paged_tokens_v2(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        int(compressed_page_size),
        max_seq_len,
        out,
        *_scratch.selection(q, max_seq_len),
    )


def _packed_kv(
    k_cache: torch.Tensor, v_cache: torch.Tensor, page_size: int
) -> torch.Tensor:
    if (
        k_cache.ndim != 4
        or v_cache.shape != k_cache.shape
        or tuple(k_cache.shape[1:]) != (page_size, 1, 256)
        or tuple(k_cache.stride()) != (page_size * 512, 512, 256, 1)
        or tuple(v_cache.stride()) != tuple(k_cache.stride())
        or v_cache.data_ptr() - k_cache.data_ptr() != 512
    ):
        raise ValueError("SYCL QSA requires adjacent packed K/V cache")
    return k_cache.as_strided(
        (k_cache.shape[0], 1, page_size, 512),
        (page_size * 512, page_size * 512, 512, 1),
    )


def sparse_paged_attention_v2(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    page_size: int,
    out: torch.Tensor,
) -> torch.Tensor:
    if int(q.shape[1]) not in (3, 6):
        raise ValueError("SYCL QSA attention supports 3 or 6 query heads")
    packed = _packed_kv(k_cache, v_cache, int(page_size))
    return _qwen38_C.token_split_attention_v3(
        q,
        packed,
        logical_indices,
        block_table,
        token_to_req,
        int(page_size),
        out,
        _scratch.attention(q),
    )


def sparse_paged_attention_q6_v1(*args: object) -> torch.Tensor:
    q = args[0]
    if not isinstance(q, torch.Tensor) or q.shape[1] != 6:
        raise ValueError("q6 entry requires six query heads")
    return sparse_paged_attention_v2(*args)


def sparse_attention_token_split_candidate_q6_v1(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    page_size: int,
    out: torch.Tensor,
    partials: torch.Tensor,
) -> torch.Tensor:
    if q.ndim != 3 or q.shape[1] != 6:
        raise ValueError("q6 token-split requires six query heads")
    return sparse_attention_token_split_candidate_v3(
        q,
        packed_kv,
        logical_indices,
        block_table,
        token_to_req,
        page_size,
        out,
        partials,
    )


def sparse_attention_token_split_candidate_v3(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    page_size: int,
    out: torch.Tensor,
    partials: torch.Tensor,
) -> torch.Tensor:
    return _qwen38_C.token_split_attention_v3(
        q,
        packed_kv,
        logical_indices,
        block_table,
        token_to_req,
        int(page_size),
        out,
        partials,
    )


def sparse_paged_attention_v3(*args: object) -> torch.Tensor:
    return sparse_paged_attention_v2(*args)


def qsa_group_compress_v2(
    raw_keys: torch.Tensor,
    raw_positions: torch.Tensor,
    compressor_state_cache: torch.Tensor,
    rope_position_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    pooled: torch.Tensor,
    first_positions: torch.Tensor,
    compress_ratio: int,
    compressed_capacity: int,
    historical_ring_proven: bool,
) -> torch.Tensor:
    return _qwen38_C.group_compress_v2(
        raw_keys,
        raw_positions,
        compressor_state_cache,
        rope_position_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        compress_ratio,
        compressed_capacity,
        historical_ring_proven,
    )


def qsa_indexer_norm_rope_v1(
    input: torch.Tensor,
    output: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    mrope: bool,
    positions_bounds_proven: bool,
) -> torch.Tensor:
    return _qwen38_C.qsa_sycl_indexer_norm_rope_v2(
        input,
        output,
        weight,
        positions,
        cos_sin_cache,
        mrope,
        False,
        positions_bounds_proven,
    )


def qsa_indexer_norm_rope_v2(
    input: torch.Tensor,
    output: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    mrope: bool,
    is_neox_style: bool,
    enable_fp32_compute: bool,
) -> torch.Tensor:
    if is_neox_style is not True or enable_fp32_compute is not False:
        raise ValueError("SYCL indexer v2 requires NeoX and FP16 products")
    # Unlike v1, this ABI has no host proof argument. The eager-FP16 native
    # route guards every cache read on-device, including negative wrapping.
    return _qwen38_C.qsa_sycl_indexer_norm_rope_v2(
        input,
        output,
        weight,
        positions,
        cos_sin_cache,
        mrope,
        True,
        False,
    )


def indexer_norm_rope_v2(
    input: torch.Tensor,
    output: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    mrope: bool,
    eager_fp16: bool,
    positions_bounds_proven: bool,
) -> torch.Tensor:
    return _qwen38_C.qsa_sycl_indexer_norm_rope_v2(
        input,
        output,
        weight,
        positions,
        cos_sin_cache,
        mrope,
        eager_fp16,
        positions_bounds_proven,
    )


def indexer_projection_int4_v1(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    group_scales: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    return _qwen38_C.qsa_sycl_indexer_projection_int4_v1(
        input,
        packed_weight,
        group_scales,
        output,
    )


def q_norm_rope_select_v1(
    projected_q: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    compressed_key_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    q_output: torch.Tensor,
    out: torch.Tensor,
    compressed_page_size: int,
    *,
    max_seq_len: int,
    mrope: bool,
    eager_fp16: bool,
    positions_bounds_proven: bool,
) -> torch.Tensor:
    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError("SYCL QSA fused composition requires host max_seq_len")
    return _qwen38_C.qsa_sycl_q_norm_rope_select_v1(
        projected_q,
        norm_weight,
        positions,
        cos_sin_cache,
        compressed_key_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        q_output,
        out,
        *_scratch.selection(projected_q, max_seq_len),
        int(compressed_page_size),
        max_seq_len,
        mrope,
        eager_fp16,
        positions_bounds_proven,
    )


def qsa_q_norm_rope_select_v1(
    projected_q: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    compressed_key_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    q_output: torch.Tensor,
    out: torch.Tensor,
    mrope: bool,
    positions_bounds_proven: bool,
    *,
    max_seq_len: int,
    compressed_page_size: int,
) -> torch.Tensor:
    """Legacy 13 positional args with trusted-host SYCL dispatch metadata."""
    if not isinstance(q_output, torch.Tensor):
        raise TypeError("q_output must be a Tensor or an empty Tensor sentinel")
    normalized_q = (
        _scratch.normalized_q(projected_q)
        if q_output.numel() == 0
        else q_output
    )
    return q_norm_rope_select_v1(
        projected_q,
        norm_weight,
        positions,
        cos_sin_cache,
        compressed_key_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        normalized_q,
        out,
        compressed_page_size,
        max_seq_len=max_seq_len,
        mrope=mrope,
        eager_fp16=False,
        positions_bounds_proven=positions_bounds_proven,
    )


def qsa_store_cache_rows_v3(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> torch.Tensor:
    return _qwen38_C.qsa_sycl_store_cache_rows_v3(
        cache,
        slot_mapping,
        rows,
        False,
    )


def qsa_store_cache_rows_r_aware_v1(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
    unique_slots_proven: bool,
) -> torch.Tensor:
    return _qwen38_C.qsa_sycl_store_cache_rows_v3(
        cache,
        slot_mapping,
        rows,
        unique_slots_proven,
    )


def try_store_m1_transaction_v1(
    stores: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    enable_fused: bool,
) -> bool:
    if type(enable_fused) is not bool:
        raise TypeError("enable_fused must be bool")
    # Probe once at import: old DSOs keep the original ordered 2/3-launch
    # transaction even when the vLLM fused flag is enabled.
    if enable_fused and _m1_fused_store is not None:
        return bool(_m1_fused_store(stores))
    return bool(_qwen38_C.qsa_sycl_try_store_m1_transaction_v1(stores))


def get_qkv_postprocess(*, mrope: bool):
    """Return a proof-gated native callable for the Qwen3 Next adapter."""
    return (
        esimd_qkv_split_norm_rope_mrope_v1
        if mrope
        else esimd_qkv_split_norm_rope
    )


get_QKV_postprocess = get_qkv_postprocess


def esimd_qkv_split_norm_rope(
    qkv: torch.Tensor,
    q: torch.Tensor,
    gate: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    norm_wq: torch.Tensor,
    norm_wk: torch.Tensor,
    positions: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    attn_output_gate: bool,
    rotary_dim: int,
    cos_sin_cache: torch.Tensor,
    *,
    positions_bounds_proven: bool = False,
) -> torch.Tensor:
    if rotary_dim != 64:
        raise ValueError("SYCL QKV supports rotary_dim=64")
    return _qwen38_C.qsa_sycl_qkv_postprocess_v1(
        qkv,
        q,
        gate,
        k,
        v,
        norm_wq,
        norm_wk,
        positions,
        cos_sin_cache,
        q_heads,
        kv_heads,
        attn_output_gate,
        False,
        positions_bounds_proven,
    )


def esimd_qkv_split_norm_rope_mrope_v1(
    qkv: torch.Tensor,
    q: torch.Tensor,
    gate: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    norm_wq: torch.Tensor,
    norm_wk: torch.Tensor,
    positions: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    attn_output_gate: bool,
    positions_bounds_proven: bool,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    return _qwen38_C.qsa_sycl_qkv_postprocess_v1(
        qkv,
        q,
        gate,
        k,
        v,
        norm_wq,
        norm_wk,
        positions,
        cos_sin_cache,
        q_heads,
        kv_heads,
        attn_output_gate,
        True,
        positions_bounds_proven,
    )
