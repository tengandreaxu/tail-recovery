#!/bin/bash

set -x;
set -e;

source exports;

## OLS

## Negative
python3 training/single_rolling.py \
    --ols-only \
    --lag \
    --otm-atm-only \
    --create-name \
    --horizon 1 \
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
    --horizon 1 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;



    

## DNN - OTM-ATM

## Negative
python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --call-only \
    --otm-atm-only \
    --create-name \
    --horizon 1 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --put-only \
    --otm-atm-only \
    --create-name \
    --horizon 1 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --otm-atm-only \
    --create-name \
    --horizon 1 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

### Positive
python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --std 2.0 \
    --call-only \
    --otm-atm-only \
    --create-name \
    --horizon 1 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --std 2.0 \
    --put-only \
    --otm-atm-only \
    --create-name \
    --horizon 1 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --std 2.0 \
    --otm-atm-only \
    --create-name \
    --horizon 1 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

## DNN - OTM ONLY

### Negative
python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --call-only \
    --otm-only \
    --create-name \
    --horizon 1 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --put-only \
    --otm-only \
    --create-name \
    --horizon 1 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --otm-only \
    --create-name \
    --horizon 1 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

### stopped here
### Positive
python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --std 2.0 \
    --call-only \
    --otm-only \
    --create-name \
    --horizon 1 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --std 2.0 \
    --put-only \
    --otm-only \
    --create-name \
    --horizon 1 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --std 2.0 \
    --otm-only \
    --create-name \
    --horizon 1 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;


# LASSO BIG - NEGATIVE
python3 training/single_rolling.py \
    --lasso-only \
    --lag \
    --otm-atm-only \
    --create-name \
    --horizon 1 \
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
    --horizon 1 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

# LASSO SMALL
python3 training/single_rolling.py \
    --lasso-only \
    --lag \
    --otm-atm-only \
    --create-name \
    --horizon 1 \
    --has-earnings \
    --data-path $GRIDS_SMALL \
    --res-dir $RES_DIR;

# NNET SMALL
python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --otm-atm-only \
    --create-name \
    --horizon 1 \
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
    --horizon 1 \
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
    --horizon 1 \
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
    --horizon 1 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --lasso-only \
    --lag \
    --put-only \
    --otm-atm-only \
    --create-name \
    --horizon 1 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;


