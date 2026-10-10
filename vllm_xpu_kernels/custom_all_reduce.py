"""Python side of the XPU one-shot custom all-reduce (TP=2, Level Zero IPC).

Usage (inside an initialized torch.distributed job):
    cpu_group = dist.new_group(backend="gloo")
    car = XpuCustomAllReduce(cpu_group, torch.device("xpu", local_rank))
    if car.can_use(x):
        car.all_reduce_(x)          # in place, on the current XPU stream
    else:
        dist.all_reduce(x, group=device_group)

All ranks of the group must issue the same sequence of all_reduce_ calls
(same sizes, same order) on their current stream, like any collective.
"""

import os
from contextlib import nullcontext

import torch
import torch.distributed as dist


def _ops():
    import vllm_xpu_kernels._xpu_C  # noqa: F401

    return torch.ops._xpu_C


def is_available() -> bool:
    try:
        return hasattr(_ops(), "custom_ar_create")
    except ImportError:
        return False


class XpuCustomAllReduce:
    _SUPPORTED = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(self, cpu_group, device, max_bytes=128 * 1024):
        self.disabled = True
        self.group = cpu_group
        self.world_size = dist.get_world_size(cpu_group)
        self.rank = dist.get_rank(cpu_group)
        if self.world_size != 2:
            return  # only TP=2 is implemented
        dev = torch.device(device)
        ops = _ops()
        self.max_bytes = max_bytes
        with torch.xpu.device(dev):
            self.handle = ops.custom_ar_create(max_bytes)
            mine = list(ops.custom_ar_export(self.handle))
            info = [None] * self.world_size
            dist.all_gather_object(info, mine, group=cpu_group)
            mask = int(os.environ.get("VLLM_XPU_CUSTOM_AR_IPC_MODES", "7"))
            self.ipc_mode = ops.custom_ar_open(
                self.handle, self.rank, info[1 - self.rank], mask
            )
        # both ranks must have imported before either may exit / free
        dist.barrier(group=cpu_group)
        self.device = dev
        self._ar = ops.custom_ar_all_reduce_
        self.disabled = False

    def can_use(self, x: torch.Tensor) -> bool:
        return (
            not self.disabled
            and x.device == self.device
            and x.dtype in self._SUPPORTED
            and x.is_contiguous()
            and x.numel() * x.element_size() <= self.max_bytes
        )

    def all_reduce_(self, x: torch.Tensor) -> torch.Tensor:
        self._ar(x, self.handle)
        return x

    def capture(self):
        """IPC buffers and device epochs need no per-graph registration."""
        return nullcontext()

    def close(self):
        if not self.disabled:
            dist.barrier(group=self.group)
            _ops().custom_ar_destroy(self.handle)
            self.disabled = True
