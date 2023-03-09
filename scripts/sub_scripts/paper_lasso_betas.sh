#!/bin/bash

set -x;
set -e;

LASSO_BETAS=rolling_lasso_betas
NEGATIVE_PUT=lag=True_horizon=5_earnings=True_moneyness=otm_only_std=-2.0_data=tickers_final_full_options=put_only_iv_type=all
NEGATIVE_CALL=lag=True_horizon=5_earnings=True_moneyness=otm_only_std=-2.0_data=tickers_final_full_options=call_only_iv_type=all

POSITIVE_PUT=lag=True_horizon=5_earnings=True_moneyness=otm_only_std=2.0_data=tickers_final_full_options=put_only_iv_type=all
POSITIVE_CALL=lag=True_horizon=5_earnings=True_moneyness=otm_only_std=2.0_data=tickers_final_full_options=call_only_iv_type=all


python3 analysis/analysis_lasso_betas.py \
    --lasso-coeff-put ${RES_DIR}${NEGATIVE_PUT}/${LASSO_BETAS} \
    --lasso-coeff-call ${RES_DIR}${NEGATIVE_CALL}/${LASSO_BETAS};


python3 analysis/analysis_lasso_betas.py \
    --lasso-coeff-put ${RES_DIR}${POSITIVE_PUT}/${LASSO_BETAS} \
    --lasso-coeff-call ${RES_DIR}${POSITIVE_CALL}/${LASSO_BETAS} \
    --is-pos;
