#!/bin/bash

source exports;
set -x;
set -e;

#NEGATIVE OTM+ATM
python3 analysis/analysis_performance_from_different_results.py \
	--res-1 $NEG_CALL_ONLY \
	--res-2 $NEG_PUT_ONLY \
	--res-3 $NEGATIVE_5_DAYS \
	--predictions-dir $RES_DIR \
	--lag \
	--t 1

#POSITIVE OTM+ATM
python3 analysis/analysis_performance_from_different_results.py \
	--res-1 $POS_CALL_ONLY \
	--res-2 $POS_PUT_ONLY \
	--res-3 $POSITIVE_5_DAYS \
	--predictions-dir $RES_DIR \
	--is-pos \
	--lag \
	--t 1


python3 analysis/analysis_performance_from_different_results.py \
	--res-1 $NEG_CALL_OTM_ONLY \
	--res-2 $NEG_PUT_OTM_ONLY \
	--res-3 $NEG_BOTH_OTM_ONLY \
	--predictions-dir $RES_DIR \
	--otm-only \
	--lag \
	--t 1

python3 analysis/analysis_performance_from_different_results.py \
	--res-1 $POS_CALL_OTM_ONLY \
	--res-2 $POS_PUT_OTM_ONLY \
	--res-3 $POS_BOTH_OTM_ONLY \
	--predictions-dir $RES_DIR \
	--otm-only \
	--is-pos \
	--lag \
	--t 1
