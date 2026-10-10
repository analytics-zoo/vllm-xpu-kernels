# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the imported GDN DSO's fixed capability snapshot."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch


OPS = ("gdn_decode_sycl", "gdn_spec_v2_sycl", "gdn_norm_gate_sycl",
       "gdn_norm_int4_sycl")


def load_provider(monkeypatch, available, dispatch=True):
    name = "_qwen38_gdn_capability_test"
    package = ModuleType(name)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, name + "._qwen38_C", ModuleType("_qwen38_C"))
    monkeypatch.delenv("QWEN38_SYCL_LIBRARY", raising=False)
    schemas = Mock(side_effect=lambda name: [object()]
                   if name.rsplit("::", 1)[1] in available else [])
    dispatch_probe = Mock(return_value=dispatch)
    monkeypatch.setattr(torch._C, "_jit_get_schemas_for_operator", schemas)
    monkeypatch.setattr(torch._C, "_dispatch_has_kernel_for_dispatch_key",
                        dispatch_probe)
    path = Path(__file__).resolve().parents[1] / "vllm_xpu_kernels/qwen38_gdn.py"
    spec = importlib.util.spec_from_file_location(name + ".qwen38_gdn", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, schemas, dispatch_probe


@pytest.mark.parametrize("dispatch", [True, False])
def test_capabilities_are_probed_once_after_dso_import(monkeypatch, dispatch):
    module, schemas, probe = load_provider(monkeypatch, OPS, dispatch)
    assert schemas.call_count == 4
    assert probe.call_count == 4
    for _ in range(36):
        assert module.has_esimd_gdn_conv_fused_seq_spec_v2() is dispatch
        assert module.has_esimd_gdn_conv_fused_seq(4, 12, 128, 128) is dispatch
        assert module.has_gdn_norm_int4_sycl() is dispatch
    assert schemas.call_count == 4
    assert probe.call_count == 4
    assert hasattr(module, "esimd_gdn_conv_fused_seq") is dispatch
    assert hasattr(module, "esimd_gdn_conv_fused_seq_spec_v2") is dispatch


def test_missing_old_dso_ops_remain_absent_without_reprobes(monkeypatch):
    module, schemas, probe = load_provider(monkeypatch, ())
    for _ in range(36):
        assert not module.has_esimd_gdn_conv_fused_seq_spec_v2()
        assert not module.has_esimd_gdn_conv_fused_seq(4, 12, 128, 128)
    assert schemas.call_count == 4
    probe.assert_not_called()
    assert not hasattr(module, "esimd_gdn_conv_fused_seq")
    assert not hasattr(module, "esimd_gdn_conv_fused_seq_spec_v2")


def test_geometry_is_not_replaced_by_cached_availability(monkeypatch):
    module, schemas, _ = load_provider(monkeypatch, OPS)
    assert module.has_esimd_gdn_conv_fused_seq(4, 12, 128, 128)
    assert module.has_esimd_gdn_conv_fused_seq(2, 6, 128, 128)
    for shape in ((4, 12, 64, 128), (4, 12, 128, 64), (8, 24, 128, 128)):
        assert not module.has_esimd_gdn_conv_fused_seq(*shape)
    assert schemas.call_count == 4
