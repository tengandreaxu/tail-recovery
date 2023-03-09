#!/bin/bash
source exports;
set -x;
set -e;

############
# Negative
############

python3 analysis/analysis_otm_average_betas.py \
	--call-only ${RES_DIR}${NEG_CALL_ONLY} \
	--put-only ${RES_DIR}${NEG_PUT_ONLY};


############
# Positive
############

python3 analysis/analysis_otm_average_betas.py \
	--call-only ${RES_DIR}${POS_CALL_ONLY} \
	--put-only ${RES_DIR}${POS_PUT_ONLY} \
	--is-pos;
