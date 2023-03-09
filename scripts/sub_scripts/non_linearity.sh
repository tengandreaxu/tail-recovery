#!/bin/bash

source exports;
set -x;
set -e;

python3 non_linearity/correlation_lasso_vs_nnet.py \
    --lasso-pos ${RES_DIR}${POSITIVE_5_DAYS}rolling_lasso_full_pred \
    --lasso-neg ${RES_DIR}${NEGATIVE_5_DAYS}rolling_lasso_full_pred \
    --nnet-neg ${RES_DIR}${NEGATIVE_5_DAYS}rolling_nnet_full_pred \
    --nnet-pos ${RES_DIR}${POSITIVE_5_DAYS}rolling_nnet_full_pred