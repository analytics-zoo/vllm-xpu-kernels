# Shared setup for the custom all-reduce tests / benchmarks.
# Run: torchrun --nproc_per_node=2 tests/custom_ar/test_custom_ar.py
import os

import torch
import torch.distributed as dist

from vllm_xpu_kernels.custom_all_reduce import XpuCustomAllReduce  # noqa: E402


def setup(max_bytes=128 * 1024):
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.xpu.set_device(local)
    # pin to the CPUs of the GPU's NUMA node (the 2 B70s hang off different
    # sockets on this box); KDEV_AR_CPUS="a-b,c-d" per rank overrides
    cpus = os.environ.get("KDEV_AR_CPUS")
    if cpus:
        spec = cpus.split(",")[local]
        lo, hi = (int(v) for v in spec.split("-"))
        os.sched_setaffinity(0, range(lo, hi + 1))
    dev = torch.device("xpu", local)
    dist.init_process_group("xccl")
    cpu = dist.new_group(backend="gloo")
    car = XpuCustomAllReduce(cpu, dev, max_bytes=max_bytes)
    assert not car.disabled
    # warm XCCL
    t = torch.ones(2048, device=dev, dtype=torch.float16)
    dist.all_reduce(t)
    torch.xpu.synchronize()
    return rank, dev, cpu, car


def log0(rank, *a):
    if rank == 0:
        print(*a, flush=True)
