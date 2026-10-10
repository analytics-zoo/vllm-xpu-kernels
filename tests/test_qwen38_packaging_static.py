# SPDX-License-Identifier: Apache-2.0
"""Build metadata checks that do not import extensions or configure CMake."""

import ast
import tomllib
from pathlib import Path

import pytest
from setuptools import Distribution, Extension, find_packages
from setuptools.command.bdist_wheel import bdist_wheel


ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = (
    "qwen38.py",
    "qwen38_gdn.py",
    "qwen38_moe.py",
    "qwen38_qsa.py",
)


def _setup_tree() -> ast.Module:
    return ast.parse((ROOT / "setup.py").read_text())


def test_mhc_build_option_is_forwarded_to_cmake():
    configure = next(
        node
        for node in ast.walk(_setup_tree())
        if isinstance(node, ast.FunctionDef) and node.name == "configure"
    )
    options = next(
        node.value
        for node in configure.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_kernel_options"
            for target in node.targets
        )
    )
    assert isinstance(options, ast.List)
    assert "MHC_KERNELS_ENABLED" in [item.value for item in options.elts]

    forward = next(
        node
        for node in configure.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "opt"
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "_kernel_options"
    )
    calls = [
        node for node in ast.walk(forward) if isinstance(node, ast.Call)
    ]
    assert any(
        isinstance(call.func, ast.Name)
        and call.func.id == "_is_enabled"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "opt"
        for call in calls
    )
    assert "option(MHC_KERNELS_ENABLED" in (ROOT / "CMakeLists.txt").read_text()


def test_four_providers_are_discovered_as_package_sources():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    discovery = metadata["tool"]["setuptools"]["packages"]["find"]
    assert "vllm_xpu_kernels" in find_packages(
        where=str(ROOT / discovery["where"][0]),
        include=discovery["include"],
    )
    for provider in PROVIDERS:
        assert (ROOT / "vllm_xpu_kernels" / provider).is_file()


def test_qwen38_cpython_abi_does_not_change_legacy_extensions():
    extension_calls = (
        node
        for node in ast.walk(_setup_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "CMakeExtension"
    )
    extensions = {
        next(kw.value.value for kw in call.keywords if kw.arg == "name"):
        {kw.arg: kw.value for kw in call.keywords if kw.arg != "name"}
        for call in extension_calls
    }
    qwen = extensions.pop("vllm_xpu_kernels._qwen38_C")
    assert isinstance(qwen["py_limited_api"], ast.Constant)
    assert qwen["py_limited_api"].value is False
    assert extensions
    assert all("py_limited_api" not in kwargs for kwargs in extensions.values())

    initializer = next(
        node
        for node in ast.walk(_setup_tree())
        if isinstance(node, ast.FunctionDef)
        and node.name == "__init__"
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "setdefault"
            and child.args
            and isinstance(child.args[0], ast.Constant)
            and child.args[0].value == "py_limited_api"
            for child in ast.walk(node)
        )
    )
    assert any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "setdefault"
        and len(child.args) == 2
        and isinstance(child.args[1], ast.Constant)
        and child.args[1].value is True
        for child in ast.walk(initializer)
    )

    cmake = (ROOT / "CMakeLists.txt").read_text()
    qwen_target = cmake.split("if(QWEN38_KERNELS_ENABLED)", 1)[1].split(
        "endif()", 1
    )[0]
    assert "WITH_SOABI" in qwen_target
    assert "USE_SABI" not in qwen_target
    assert "LIBRARIES ${QWEN38_TORCH_PYTHON_LIBRARY}" in qwen_target


def _wheel_command_class():
    # Execute only the command class, never setup.py's SCM/build/install code.
    definition = next(
        node for node in _setup_tree().body
        if isinstance(node, ast.ClassDef)
        and node.name == "qwen38_bdist_wheel"
    )
    namespace = {"bdist_wheel": bdist_wheel, "Path": Path}
    tree = ast.Module(body=[definition], type_ignores=[])
    exec(compile(tree, "setup.py", "exec"), namespace)
    return namespace[definition.name]


@pytest.mark.parametrize("limited", [None, "cp38", "cp39"])
@pytest.mark.parametrize("payload", ["source", "mixed", "precompiled",
                                     "precompiled_only"])
def test_qwen38_wheel_cannot_claim_abi3(limited, payload):
    legacy = Extension("vllm_xpu_kernels._C", [], py_limited_api=True)
    qwen = Extension("vllm_xpu_kernels._qwen38_C", [],
                     py_limited_api=False)
    extensions = {
        "source": [qwen], "mixed": [qwen, legacy],
        "precompiled": [legacy], "precompiled_only": [],
    }[payload]
    data = ({"vllm_xpu_kernels": [
        "_qwen38_C.cpython-312-x86_64-linux-gnu.so"]}
        if payload.startswith("precompiled") else {})
    distribution = Distribution({"name": "wheel-tag-probe", "version": "1.0",
                                 "ext_modules": extensions,
                                 "package_data": data})
    command = _wheel_command_class()(distribution)
    command.plat_name = "linux_x86_64"
    command.py_limited_api = limited
    command.finalize_options()
    tag = command.get_tag()
    assert tag[:2] == bdist_wheel_tag_for_current_python()
    assert command.root_is_pure is False
    assert command.py_limited_api is False
    assert command.get_tag() == tag  # Repeated metadata/filename queries.
    assert legacy.py_limited_api is True
    assert qwen.py_limited_api is False


def bdist_wheel_tag_for_current_python():
    distribution = Distribution({"name": "wheel-tag-probe", "version": "1.0",
                                 "ext_modules": [Extension("probe", [])]})
    command = bdist_wheel(distribution)
    command.plat_name = "linux_x86_64"
    command.root_is_pure = False
    return command.get_tag()[:2]


@pytest.mark.parametrize("limited", [None, "cp38", "cp39"])
def test_legacy_only_wheel_keeps_upstream_tag_policy(limited):
    distribution = Distribution({"name": "wheel-tag-probe", "version": "1.0",
        "ext_modules": [
        Extension("vllm_xpu_kernels._C", [], py_limited_api=True)],
        "package_data": {"vllm_xpu_kernels": ["_C.abi3.so"]}})
    commands = [kind(distribution)
                for kind in (_wheel_command_class(), bdist_wheel)]
    for command in commands:
        command.plat_name = "linux_x86_64"
        command.py_limited_api = limited
        command.finalize_options()
    assert commands[0].get_tag() == commands[1].get_tag()
    assert commands[0].py_limited_api == limited


def test_all_wheel_frontends_share_the_registered_command():
    registration = next(
        node.value for node in _setup_tree().body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "cmdclass"
                for t in node.targets)
    )
    assert isinstance(registration, ast.Dict)
    commands = dict(zip((key.value for key in registration.keys),
                        registration.values))
    assert commands["bdist_wheel"].id == "qwen38_bdist_wheel"
    setup_call = next(
        node for node in ast.walk(_setup_tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    assert any(kw.arg == "cmdclass" and kw.value.id == "cmdclass"
               for kw in setup_call.keywords)
    for filename in (".github/workflows/wheel-per-commit.yaml",
                     "build_script/build_wheel_docker.sh"):
        source = (ROOT / filename).read_text()
        assert "setup.py bdist_wheel" in source
        assert "--py-limited-api=cp38" in source  # Legacy-only policy retained.
        assert "only without _qwen38_C" in source
