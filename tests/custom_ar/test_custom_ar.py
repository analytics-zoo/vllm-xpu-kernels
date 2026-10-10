# Correctness of the custom all-reduce vs XCCL (bitwise), and rank agreement.
# Many calls are queued back-to-back without any host sync.
import random

import torch
import torch.distributed as dist

from common import log0, setup

MAX_BYTES = 256 * 1024
rank, dev, cpu, car = setup(MAX_BYTES)
torch.manual_seed(1234 + rank)
random.seed(99)  # same size sequence on both ranks

SIZES = [
    1,
    2,
    3,
    7,
    8,
    9,
    15,
    16,
    17,
    100,
    1000,
    2047,
    2048,
    2049,
    4096,
    8191,
    8192,
    16384,
    16385,
    32768,
    65535,
    65536,
]
fails = 0


def run_batch(tensors):
    refs = [t.clone() for t in tensors]
    for t in tensors:  # back-to-back, no sync
        car.all_reduce_(t)
    for r in refs:
        dist.all_reduce(r)
    torch.xpu.synchronize()
    return tensors, refs


def bitwise_equal(a, b):
    return torch.equal(
        a.view(torch.int16) if a.element_size() == 2 else a.view(torch.int32),
        b.view(torch.int16) if b.element_size() == 2 else b.view(torch.int32),
    )


# 1) each dtype x size, 20 iterations back-to-back
for dt in (torch.float16, torch.bfloat16, torch.float32):
    ts = []
    for n in SIZES:
        if n * torch.tensor([], dtype=dt).element_size() > MAX_BYTES:
            continue
        ts += [torch.randn(n, device=dev, dtype=dt) * 3 for _ in range(20)]
    outs, refs = run_batch(ts)
    bad = sum(int(not bitwise_equal(o, r)) for o, r in zip(outs, refs))
    hs = [None, None]
    dist.all_gather_object(
        hs,
        [
            o.view(-1)[:64].float().sum().item() + o.float().sum().item()
            for o in outs
        ],
        group=cpu,
    )
    ok = bad == 0 and hs[0] == hs[1]
    fails += 0 if ok else 1
    log0(
        rank,
        f"{'OK  ' if ok else 'FAIL'} {dt} sizes 1..64K x20: "
    f"{len(outs)} calls, bitwise mismatches vs XCCL={bad}, "
    f"ranks agree={hs[0] == hs[1]}",
    )

# 2) misaligned (offset-1 views -> scalar path), 2D shapes
base = [torch.randn(4097, device=dev, dtype=torch.float16) for _ in range(50)]
ts = [b[1:] for b in base] + [
    torch.randn(8, 2048, device=dev, dtype=torch.float16) for _ in range(50)
]
outs, refs = run_batch(ts)
bad = sum(
    int(not bitwise_equal(o.contiguous(), r.contiguous()))
    for o, r in zip(outs, refs)
)
fails += int(bad != 0)
log0(
    rank,
    f"{'OK  ' if bad == 0 else 'FAIL'} misaligned/2D: "
    f"{len(outs)} calls, mismatches={bad}",
)

# 3) 3000 random-size calls interleaved with other GPU work
ns = [
    random.choice([2048, 2048, 4096, 16384, random.randint(1, 65536)])
    for _ in range(3000)
]
ts = [torch.randn(n, device=dev, dtype=torch.float16) for n in ns]
refs = [t.clone() for t in ts]
junk = torch.randn(1024, 1024, device=dev, dtype=torch.float16)
for i, t in enumerate(ts):
    if i % 7 == 0:
        junk = junk @ junk * 0.001  # other queued work between calls
    car.all_reduce_(t)
for r in refs:
    dist.all_reduce(r)
torch.xpu.synchronize()
bad = sum(int(not bitwise_equal(o, r)) for o, r in zip(ts, refs))
hs = [None, None]
dist.all_gather_object(hs, [o.float().sum().item() for o in ts], group=cpu)
ok = bad == 0 and hs[0] == hs[1]
fails += 0 if ok else 1
log0(
    rank,
    f"{'OK  ' if ok else 'FAIL'} random-size stress: "
    f"{len(ts)} calls, mismatches={bad}, ranks agree={hs[0] == hs[1]}",
)

# 4) long run of the decode shape, 20000 calls in place on the same tensor chain
x = torch.randn(1, 2048, device=dev, dtype=torch.float16) * 1e-3
y = x.clone()
for i in range(20000):
    car.all_reduce_(x)
    x.mul_(0.5)
for i in range(20000):
    dist.all_reduce(y)
    y.mul_(0.5)
torch.xpu.synchronize()
ok = bitwise_equal(x, y)
fails += int(not ok)
log0(
    rank,
    f"{'OK  ' if ok else 'FAIL'} 20000 chained in-place [1,2048] fp16 calls",
)

# 5) skewed ranks: one rank's GPU (and host) is busy / late before some calls
import time  # noqa: E402

big = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
ts = [
    torch.randn(
        random.choice([2048, 16384, 65536]), device=dev, dtype=torch.float16
    )
    for _ in range(300)
]
refs = [t.clone() for t in ts]
for i, t in enumerate(ts):
    if i % 10 == 0 and rank == (i // 10) % 2:
        big = big @ big * 1e-4  # this rank's GPU is late
    if i % 50 == 0 and rank == 1:
        time.sleep(0.01)  # this rank's host is late
    car.all_reduce_(t)
for r in refs:
    dist.all_reduce(r)
torch.xpu.synchronize()
bad = sum(int(not bitwise_equal(o, r)) for o, r in zip(ts, refs))
fails += int(bad != 0)
log0(
    rank,
    f"{'OK  ' if bad == 0 else 'FAIL'} skewed ranks (GPU/host late): "
    f"{len(ts)} calls, mismatches={bad}",
)

err = torch.ops._xpu_C.custom_ar_error(car.handle)
log0(rank, f"spin-timeout flag = {err}")
log0(rank, "ALL PASS" if fails == 0 and err == 0 else f"{fails} FAILURES")
car.close()
dist.destroy_process_group()
