#!/bin/bash

source exports;
RES_TABLES=$RES_FOLDER'/tables/'
OTM_NEGATIVE_SMALL='lag=True_horizon=5_earnings=True_moneyness=otm_atm_only_std=-2.0_data=tickers_small_post_options=all_iv_type=all'

OTM_POSITIVE_R2=${RES_DIR}${POSITIVE_5_DAYS}'/r2'
OTM_NEGATIVE_R2=${RES_DIR}${NEGATIVE_5_DAYS}'/r2'

OTM_NEGATIVE_R2_RESULT=$RES_FOLDER'/backward_looking/negative/'
OTM_POSITIVE_R2_RESULT=$RES_FOLDER'/backward_looking/positive/'

set -x;
set -e;

# All results folder
mkdir -p $RES_FOLDER;

# Tables section
mkdir -p $RES_TABLES;

# This script will reproduce all paper figures

while getopts ":h:help:" option; do
    case $option in
        help) Help; exit;;
        h) HOST=1;;
        \?) echo "Error: Invalid option"
            exit;;
    esac
done
echo ${HOST};

# Figure 1 and Table 6 Bins count
./scripts/sub_scripts/counting_results.sh;

# Figure 2 - Distributions
## Positive
./scripts/sub_scripts/paper_distributions.sh

#Figure 3-4 - BACKWARD Looking
./scripts/sub_scripts/paper_multiple_analysis.sh


# Figure 5-6-7 Forward Looking OTM_ATM and OTM only
./scripts/sub_scripts/paper_multiple_analysis_nnet_otm_weekly.sh

# Figure 8 - OTM puts and Calls average betas
./scripts/sub_scripts/paper_lasso_betas.sh

# Figure 9 - OTM volume across time and maturity
./scripts/paper_otm_puts_calls_volume.sh;
cp -r res_common/volume_count $RES_FOLDER;
#Figure 10
cp ${RES_DIR}${NEGATIVE_5_DAYS}'/r2/lasso_vs_nnet_rolling_Mean_R2.png' ${RES_FOLDER}/'lasso_vs_nnet_big.png'
cp ${RES_DIR}${NEGATIVE_5_DAYS_SMALL}'/r2/lasso_vs_nnet_rolling_Mean_R2.png' $RES_FOLDER'/lasso_vs_nnet_small.png'

# Figure 11 - Big vs Small
./scripts/sub_scripts/big_predicting_small_subset.sh;

# Figure 12 - 14 Earnings Event Study
./scripts/paper_earnings_event_study.sh;
cp -r res_common/event_studies/jumps $RES_FOLDER;

# Figure 13 - 15 Scatter Plots
./scripts/paper_scatter_plot_on_earnings.sh;
mv earnings $RES_FOLDER;

# Figure 16 Covariate Importance
# Figure 17 Non Linearity
./scripts/non_linearity.sh


tar czvf res_paper.tar.gz $RES_FOLDER;
mv res_paper.tar.gz ~;
rm -r $RES_FOLDER;
