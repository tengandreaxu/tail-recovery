#!/bin/bash

set -x;
set -e;
source exports;

DATA_FOLDER=${DATASET}'raw_option_data/'
OPTIONS_FOLDER=${DATASET}grids/number_of_options_per_group
TICKERS_SUBSET=${DATASET_SMALL}tickers_used_for_training.csv
TICKERS_ALL=${DATASET}tickers_used_for_training.csv

DATA_FOLDER_SMALL=${DATASET_SMALL}/raw_option_data/

### BINS ###
python3 data_statistics/count_number_of_options_per_bins.py \
	--data-path $OPTIONS_FOLDER \
	--is-big;

python3 data_statistics/count_number_of_options_per_bins.py \
	--data-path $OPTIONS_FOLDER \
	--tickers-subset $TICKERS_SUBSET;

### put call volume ###
python3 data_statistics/count_put_call_volume_over_years.py \
	--data-path $DATA_FOLDER \
	--is-big \
	--tickers-used $TICKERS_ALL;

python3 data_statistics/count_put_call_volume_over_years.py \
	--data-path $DATA_FOLDER \
	--tickers-used $TICKERS_SUBSET;

### daily number of options ###

python3 data_statistics/count_total_number_of_options.py \
	--data-path $DATA_FOLDER \
	--is-big;

python3 data_statistics/count_total_number_of_options.py \
	--data-path ${DATA_FOLDER_SMALL};

### count tickers ###
python3 data_statistics/count_number_of_tickers.py \
	--train-ticker-file ${DATASET}'grids/X_train/train_ticker.p' \
	--is-big;
python3 data_statistics/count_number_of_tickers.py \
	--train-ticker-file ${DATASET_SMALL}/grids/X_train/train_ticker.p;
