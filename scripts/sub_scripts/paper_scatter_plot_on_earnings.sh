#!/bin/bash

source exports;
set -x;
set -e;
OTM_POSITIVE_RESULT=${RES_DIR}${POSITIVE_5_DAYS}scatter
OTM_NEGATIVE_RESULT=${RES_DIR}${NEGATIVE_5_DAYS}scatter

python3 event_studies/plots_forecasted_mean.py \
	--predictions-folder ${RES_DIR}${NEGATIVE_5_DAYS} \
	--earnings-file ${FILE_EARNINGS};


python3 event_studies/plots_forecasted_mean.py \
	--predictions-folder ${RES_DIR}${POSITIVE_5_DAYS} \
	--earnings-file ${FILE_EARNINGS} \
	--is-pos;

mkdir -p paper/res_paper/earnings
cp -r $OTM_NEGATIVE_RESULT  paper/res_paper/earnings/neg
cp -r $OTM_POSITIVE_RESULT  paper/res_paper/earnings/pos

