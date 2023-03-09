#!/bin/bash

set -x;
set -e;
source exports;

OTM_POSITIVE_R2=${RES_DIR}${POSITIVE_5_DAYS}'/r2'
OTM_NEGATIVE_R2=${RES_DIR}${NEGATIVE_5_DAYS}'/r2'

OTM_NEGATIVE_R2_RESULT=$RES_FOLDER'/backward_looking/negative/'
OTM_POSITIVE_R2_RESULT=$RES_FOLDER'/backward_looking/positive/'
# Reproduce Backward and Forward Looking OOS results
python3 analysis/analysis_rolling.py\
    --name ${POSITIVE_5_DAYS} \
    --lag \
    --is-pos;

python3 analysis/analysis_rolling.py \
    --name ${NEGATIVE_5_DAYS} \
    --lag;

python3 analysis/analysis_rolling.py \
    --name ${NEGATIVE_5_DAYS_SMALL} \
    --lag \
    --is-small;


mkdir -p $OTM_POSITIVE_R2_RESULT
mkdir -p $OTM_NEGATIVE_R2_RESULT
cp -r $OTM_POSITIVE_R2 $OTM_POSITIVE_R2_RESULT
cp -r $OTM_NEGATIVE_R2 $OTM_NEGATIVE_R2_RESULT