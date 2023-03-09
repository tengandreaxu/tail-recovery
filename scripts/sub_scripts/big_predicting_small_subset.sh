#!/bin/bash

source exports;
set -x;
set -e;
BIG_OTM=${RES_DIR}${NEGATIVE_5_DAYS}rolling_pred/

SMALL_OTM=${RES_DIR}${NEGATIVE_5_DAYS_SMALL}rolling_pred/

#### Comparison between CD-NNET trained on full dataset predicting
#### extreme negative returns on firms which belong to small subset only
#### and CD-NNET trained on small dataset predicting extreme negative returns
python3 analysis/analysis_rolling_tickers_subset.py \
	--predictions-folder $BIG_OTM \
	--tickers-subset $DATASET_SMALLtickers_used_for_training.csv \
	--small-predictions-folder $SMALL_OTM
