#!/bin/bash

source exports;
set -x;
set -e;
# **************
# creates the benchmark files
# ***************

python3 benchmark/benchmark_exgdp_full.py;

python3 benchmark/benchmark_exgdp_full.py \
    --is-small \
    --data-path ${GRIDS_SMALL};