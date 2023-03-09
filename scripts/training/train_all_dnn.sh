## DNN - OTM-ATM

## Negative
python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --call-only \
    --otm-atm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --put-only \
    --otm-atm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --otm-atm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

### Positive
python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --std 2.0 \
    --call-only \
    --otm-atm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --std 2.0 \
    --put-only \
    --otm-atm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --std 2.0 \
    --otm-atm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

## DNN - OTM ONLY

### Negative
python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --call-only \
    --otm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --put-only \
    --otm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --otm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

### Positive
python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --std 2.0 \
    --call-only \
    --otm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --std 2.0 \
    --put-only \
    --otm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;

python3 training/single_rolling.py \
    --nnet-only \
    --lag \
    --std 2.0 \
    --otm-only \
    --create-name \
    --horizon 5 \
    --has-earnings \
    --data-path $GRIDS_FULL \
    --res-dir $RES_DIR;
