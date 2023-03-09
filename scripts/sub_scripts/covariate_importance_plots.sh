#!/bin/bash

source exports;
set -x;
set -e;

python3 covariates/plot_covariate_importance.py \
    --covariate-r2 ${RES_DIR}${POSITIVE_5_DAYS}covariates_r2.csv \
    --full \
    --positive;


python3 covariates/plot_covariate_importance.py \
    --covariate-r2 ${RES_DIR}${NEGATIVE_5_DAYS}covariates_r2.csv \
    --full;

python3 covariates/plot_covariate_importance.py \
    --covariate-r2 ${RES_DIR}${POSITIVE_5_DAYS}covariates_r2.csv \
    --positive;


python3 covariates/plot_covariate_importance.py \
    --covariate-r2 ${RES_DIR}${NEGATIVE_5_DAYS}covariates_r2.csv;