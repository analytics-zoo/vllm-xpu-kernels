#!/bin/bash
# Multi-process tests of the TP=2 custom all-reduce (needs 2 XPUs with P2P):
#   bash tests/custom_ar/run.sh test_custom_ar.py
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1}
cd "$(dirname "$0")"
exec torchrun --nproc_per_node 2 --master_port ${PORT:-29531} "$@"
