#!/bin/bash
set -x;
set -e;
source exports;

python3 create_datasets/create_grids.py \
    --option-data-folder ${DATASET}raw_option_data/ \
    --underlyings-info-folder ${DATASET}/underlyings_info \
    --business-days-file ${DATASET}/underlyings_info/business_days_to_maturity.csv;

python3 create_datasets/create_test_and_train_sets.py \
    --data-folder ${DATASET}/grids \
    --save-aggregate;

python3 create_datasets/create_small_datasets_from_big_grids.py \
    --big-grids ${DATASET}grids \
    --tickers-subset ${DATASET_SMALL}tickers_used_for_training.csv \
    --raw-option ${DATASET}raw_option_data \
    --grids-destination ${GRIDS_SMALL} \
    --raw-option-destination ${FOLDER_RAW_OPTION_SMALL};

python3 create_datasets/create_test_and_train_sets.py \
    --data-folder ${DATASET_SMALL}/grids \
    --save-aggregate;