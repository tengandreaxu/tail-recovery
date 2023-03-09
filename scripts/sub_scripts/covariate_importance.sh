#!/bin/bash

source exports;
set -x;
set -e;

python3 covariates/covariate_importance.py \
    --res-folder ${RES_DIR}${NEGATIVE_5_DAYS} \
    --name ${NEGATIVE_5_DAYS} \
    --data-path ${FOLDER_DATASET};

python3 covariates/covariate_importance.py \
    --res-folder ${RES_DIR}${POSITIVE_5_DAYS} \
    --name ${POSITIVE_5_DAYS} \
    --std 2.0 \
    --data-path ${FOLDER_DATASET};
