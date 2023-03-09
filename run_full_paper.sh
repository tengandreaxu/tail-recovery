#!/bin/bash

set -x;
set -e;

# ***************
# Before running make sure of the followings:
# 1. the paths in the exports file is set correctly
# 2. the server has enough free RAM: ~100Gb
# 3. the server has enough free space for the results: ~50Gb
# ***************

# 1. Create datasets
./scripts/create_datasets.sh;

# 2. Train models
./scripts/train_all.sh;

# 3. Create benchmarks
./scripts/benchmark.sh;

# 4. Figures and Tables
./scripts/reproduce_figures.sh;
