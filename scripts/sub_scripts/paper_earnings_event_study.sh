#!/bin/bash

source exports;

python3 event_studies/event_study_earnings_announcements.py \
	--full-predictions ${RES_DIR}${NEGATIVE_5_DAYS}rolling_nnet_full_pred/  \
	--earnings-announcements-file ${FILE_EARNINGS} \
	--parallel \
	--range-min -30 \
	--range-max 31 \
	--t 5;

python3 event_studies/event_study_earnings_announcements.py \
	--full-predictions ${RES_DIR}${POSITIVE_5_DAYS}rolling_nnet_full_pred/ \
	--earnings-announcements-file ${FILE_EARNINGS} \
	--parallel \
	--t 5 \
	--range-min -30 \
	--range-max 31 \
	--is-pos;
