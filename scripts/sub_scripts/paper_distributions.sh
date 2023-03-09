#!/bin/bash

set -x;
source exports;

# positive
python3 analysis/fit_exgpd.py \
	--full-dataset-prediction ${FOLDER_POSITIVE_5_DAYS}/rolling_pred/ \
	--is-pos;

python3 analysis/fit_exgpd.py \
	--full-dataset-prediction ${FOLDER_NEGATIVE_5_DAYS}/rolling_pred/
