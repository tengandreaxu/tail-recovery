#!/bin/bash

set -x;
set -e;

source exports;

## OLS

# Negative
python3 training/single_rolling.py \
    --ols-only \
    --lag \
    --otm-atm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;


### Positive
python3 training/single_rolling.py \
    --ols-only \
    --lag \
    --otm-atm-only \
    --std 2.0 \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

./scripts/training/train_all_dnn.sh

# LASSO BIG - NEGATIVE
python3 training/single_rolling.py \
    --lasso-only \
    --lag \
    --otm-atm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

# LASSO BIG - POSITIVE
python3 training/single_rolling.py \
    --lasso-only \
    --lag \
    --std 2.0 \
    --otm-atm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

# LASSO SMALL
python3 training/single_rolling.py \
    --lasso-only \
    --lag \
    --otm-atm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_SMALL \
    --res-dir $RES_DIR;

# NNET SMALL
python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --otm-atm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_SMALL \
    --res-dir $RES_DIR;

#*********************
# OLS with Lasso Parameters
#**********************

# Positive
python3 training/single_rolling.py \
    --lasso-only \
    --call-only \
    --lag \
    --otm-atm-only \
    --std 2.0 \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --lasso-only \
    --lag \
    --put-only \
    --otm-atm-only \
    --std 2.0 \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;


# Negative
python3 training/single_rolling.py \
    --lasso-only \
    --lag \
    --call-only \
    --otm-atm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --lasso-only \
    --lag \
    --put-only \
    --otm-atm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;


#*********************
# Lasso Parameters
#**********************

# Positive
python3 training/single_rolling.py \
    --lasso-only \
    --call-only \
    --lag \
    --otm-only \
    --std 2.0 \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --lasso-only \
    --lag \
    --put-only \
    --otm-only \
    --std 2.0 \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;


# Negative
python3 training/single_rolling.py \
    --lasso-only \
    --lag \
    --call-only \
    --otm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --lasso-only \
    --lag \
    --put-only \
    --otm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;


