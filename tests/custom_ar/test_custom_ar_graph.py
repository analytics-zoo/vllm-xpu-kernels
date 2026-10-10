"""Check BF16 collectives across graph replay with changing inputs."""

import torch
import torch.distributed as dist

from common import log0, setup

rank, device, cpu, car = setup()
torch.manual_seed(121 + rank)
static = torch.empty((1, 2048), device=device, dtype=torch.bfloat16)
for _ in range(3):
    static.fill_(rank + 1)
    car.all_reduce_(static)
torch.xpu.synchronize()
dist.barrier(group=cpu)
graph = torch.xpu.XPUGraph()
with car.capture(), torch.xpu.graph(graph):
    for _ in range(81):
        car.all_reduce_(static)
        static.mul_(0.5)
torch.xpu.synchronize()
for iteration in range(100):
    source = torch.randn_like(static) * 0.01
    static.copy_(source)
    reference = source.clone()
    graph.replay()
    for _ in range(81):
        dist.all_reduce(reference)
        reference.mul_(0.5)
    torch.xpu.synchronize()
    assert torch.equal(static, reference), (rank, iteration)
assert torch.ops._xpu_C.custom_ar_error(car.handle) == 0
log0(
    rank,
    "GRAPH ALL PASS: BF16 [1,2048], 81 calls/replay, 100 changed-input replays",
)
car.close()
dist.destroy_process_group()
