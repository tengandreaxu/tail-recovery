#!/bin/bash

source exports;
set -x;
set -e;

python3 data_statistics/count_put_call_volume_over_years.py \
	--data-path $FOLDER_RAW_OPTION \
	--is-big \
	--otm-only \
	--business-days-file $FILE_BDAYS \
	--stock-price-history $FILE_PRICE_HISTORY \
	--dividends $FILE_DIVIDENDS \
	--tickers-used $FILE_BIG_TICKERS_SET  \
	--parallel;


python3 data_statistics/count_put_call_volume_over_years_for_maturity.py \
	--data-path $FOLDER_RAW_OPTION \
	--is-big \
	--parallel \
	--business-days-file $FILE_BDAYS \
	--tickers-used $FILE_BIG_TICKERS_SET;
