#!/bin/bash

source exports;
set -x;
set -e;

COVARIATE_RESULTS_POS=${RES_DIR}${POSITIVE_5_DAYS}covariates_importance/
COVARIATE_RESULTS_NEG=${RES_DIR}${NEGATIVE_5_DAYS}covariates_importance/


ROLLING_PRED_POS=${RES_DIR}${POSITIVE_5_DAYS}rolling_pred/
ROLLING_PRED_NEG=${RES_DIR}${NEGATIVE_5_DAYS}rolling_pred/


python3 covariates/covariate_importance_r2.py \
    --covariate-results ${COVARIATE_RESULTS_NEG} \
    --rolling-pred ${ROLLING_PRED_NEG};


python3 covariates/covariate_importance_r2.py \
    --covariate-results ${COVARIATE_RESULTS_POS} \
    --rolling-pred ${ROLLING_PRED_POS};
    