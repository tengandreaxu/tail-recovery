#!/bin/bash

source exports;
set -x;
set -e;

LASSO_POS=${FOLDER_POSITIVE_5_DAYS}rolling_lasso_full_pred/
LASSO_NEG=${FOLDER_NEGATIVE_5_DAYS}rolling_lasso_full_pred/
NNET_POS=${FOLDER_POSITIVE_5_DAYS}rolling_nnet_full_pred/
NNET_NEG=${FOLDER_NEGATIVE_5_DAYS}rolling_nnet_full_pred/

python3 correlation_lasso_vs_nnet.py \
				--lasso-pos ${LASSO_POS} \
				--lasso-neg ${LASSO_NEG} \
				--nnet-pos ${NNET_POS} \
				--nnet-neg ${NNET_NEG}


cp -r res_common/non_linearity ./res_paper_nnet
