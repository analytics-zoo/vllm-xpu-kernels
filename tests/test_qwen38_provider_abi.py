# SPDX-License-Identifier: Apache-2.0
"""CPU 侧验证旧 adapter 的返回值契约，而非只检查输出写入。"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def provider(monkeypatch):
    # 只测试源码 Python ABI，不让源码包遮蔽正常安装的 native DSO。
    name = "_qwen38_provider_abi"
    package = ModuleType(name)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(
        sys.modules, f"{name}._qwen38_C", ModuleType("_qwen38_C")
    )
    path = Path(__file__).resolve().parents[1] / "vllm_xpu_kernels/qwen38.py"
    spec = importlib.util.spec_from_file_location(f"{name}.qwen38", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name",
    [
        "int4_linear",
        "esimd_gemv_int4",
        "esimd_gemm_int4_pgrp",
        "esimd_gemm_int4_small_n_v1",
    ],
)
def test_int4_provider_returns_exact_caller_owned_output(
    monkeypatch, provider, name
):
    native = Mock(return_value=None)
    monkeypatch.setattr(
        torch.ops._qwen38_C, "int4_linear", native, raising=False
    )
    x = torch.empty(1, 128, dtype=torch.float16)
    weight = torch.empty(16, 64, dtype=torch.uint8)
    scale = torch.empty(16, 1, dtype=torch.float16)
    output = torch.empty(1, 16, dtype=torch.float16)
    assert getattr(provider, name)(x, weight, scale, output) is output
    native.assert_called_once_with(x, weight, scale, output)


@pytest.mark.parametrize(
    "name", ["int4_linear_fused2", "esimd_gemv_int4_fused2"]
)
def test_int4_fused_provider_returns_first_output_like_legacy(
    monkeypatch, provider, name
):
    native = Mock(return_value=None)
    monkeypatch.setattr(
        torch.ops._qwen38_C, "int4_linear_fused2", native, raising=False
    )
    x = torch.empty(1, 128, dtype=torch.float16)
    weight = torch.empty(16, 64, dtype=torch.uint8)
    scale = torch.empty(16, 1, dtype=torch.float16)
    first = torch.empty(1, 16, dtype=torch.float16)
    second = torch.empty_like(first)
    args = (x, weight, scale, first, weight, scale, second)
    assert getattr(provider, name)(*args) is first
    native.assert_called_once_with(*args)
