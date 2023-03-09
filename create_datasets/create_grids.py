import os
import argparse
import pandas as pd
import numpy as np

import time
from create_datasets.constants import (
    STOCK_PRICES,
    DIVIDENDS,
    EARNINGS,
    MATURITY_BUCKETS,
    RETURNS_FOR_STATISTICS,
    DO_NOT_NORMALIZE,
)
from create_datasets.DataPreprocessor import DataPreprocessor
import multiprocessing as mp
from create_datasets.create_bdays_to_maturity import business_days_between
from util.file_handling import append_to_file
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Loader")

MONEYNESS = [-2.0, -1.0, 0.0, 1.0, 2.0]
GROUP_DISCRIMINATORS = ["trade_date", "in_moneyness", "option_type", "maturity_bucket"]

LOG_FOLDER = os.path.join(os.getcwd(), "logs")


def create_directory(path: str, name: str):
    os.makedirs(f"{path}/{name}", exist_ok=True)


def split_put_call_info(options: pd.DataFrame) -> pd.DataFrame:
    """
    This function will split a single orats option row, which contains both put and ask info
    into two rows, one row containing put info, the second one containing call info

    columns: ['strike', 'bid', 'ask', 'close', 'bidsize', 'asksize', 'volume', 'underlying', 'open_interest', 'option_type', 'orats_iv', 'expiration_date']
    """
    try:
        calls_frame = options[
            [
                "strike",
                "cbidpx",
                "caskpx",
                "cvolu",
                "coi",
                "stkpx",
                "expirdate",
                "number_days",
                "trade_date",
                "ticker",
                "caskiv",
                "cbidiv",
            ]
        ].copy()
        calls_frame.columns = [
            "strike",
            "bid",
            "ask",
            "volume",
            "open_interest",
            "underlying",
            "expiration_date",
            "number_days",
            "trade_date",
            "ticker",
            "caskiv",
            "cbidiv",
        ]
        calls_frame["option_type"] = 1
        puts_frame = options[
            [
                "strike",
                "pbidpx",
                "paskpx",
                "pvolu",
                "poi",
                "stkpx",
                "expirdate",
                "number_days",
                "trade_date",
                "ticker",
                "paskiv",
                "pbidiv",
            ]
        ].copy()
        puts_frame.columns = [
            "strike",
            "bid",
            "ask",
            "volume",
            "open_interest",
            "underlying",
            "expiration_date",
            "number_days",
            "trade_date",
            "ticker",
            "paskiv",
            "pbidiv",
        ]
        puts_frame["option_type"] = -1

        option_data = calls_frame.append(puts_frame, ignore_index=True, sort=False)
    except:
        append_to_file(f"{LOG_FOLDER}stk_missing.log", [options.iloc[0].ticker])
        return pd.DataFrame()
    return option_data


def add_maturity_bucket(options: pd.DataFrame) -> pd.DataFrame:
    """
    :param, the options DataFrame
    """
    days_to_maturity = "business_days_to_maturity"

    for i in range(len(MATURITY_BUCKETS) - 1):
        lower_maturity = MATURITY_BUCKETS[i]
        upper_maturity = MATURITY_BUCKETS[i + 1]

        options.loc[
            (options[days_to_maturity] > lower_maturity)
            & (options[days_to_maturity] <= upper_maturity),
            "maturity_bucket",
        ] = lower_maturity
        options.maturity_bucket = options.maturity_bucket.fillna(value=250)
    return options


def set_nan_iv_to_zero(options: pd.DataFrame) -> pd.DataFrame:
    """
    We retrieve ready made orats iv's
    some of them are NaN
    we set them to zero
    """

    options.paskiv = options.paskiv.fillna(0)
    options.caskiv = options.caskiv.fillna(0)

    options.pbidiv = options.pbidiv.fillna(0)
    options.cbidiv = options.cbidiv.fillna(0)
    return options


def load_option_data(data_folder: str, option_data: str) -> pd.DataFrame:
    """
    :param data_folder, the to option data folder
    :param option_data, the option data csv to be loaded
    """
    df_options = pd.read_csv(os.path.join(data_folder, option_data))
    logging.info(f"Total options rows: {df_options.shape[0]}")
    df_options["time"] = pd.to_datetime(df_options.trade_date).dt.date
    df_options = df_options[
        df_options.time <= datetime.fromisoformat("2020-09-28").date()
    ]
    df_options.pop("time")
    ticker = option_data.replace(".csv", "")
    if df_options.empty:
        logging.warning(f"ticker = {ticker} has no options!")
    df_options.index = df_options.trade_date
    df_options.index.names = ["index"]

    # ORATS column for "years to expiration"
    df_options["number_days"] = df_options.yte * 365
    df_options.number_days = df_options.number_days.astype(int)

    df_options = split_put_call_info(df_options)
    if df_options.empty:
        return pd.DataFrame()
    df_options = set_nan_iv_to_zero(df_options)
    return df_options


def load_earnings(underlyings_info_folder: str) -> pd.DataFrame:
    """
    :param underlyings_info_folder, the path to dividends, earnings, stock_price_history data

    Load and clean earnings data
    """
    earnings = pd.read_csv(os.path.join(underlyings_info_folder, EARNINGS))
    # we only need the date, ticker, and the flag
    earnings["earnings_flag"] = 1
    earnings = earnings[~earnings.earnDate.isna()]
    earnings.index = earnings.earnDate
    earnings.index.names = ["index"]
    earnings = earnings.drop(columns=["anncTod"])
    logger.info(f"Loaded Earnings")
    return earnings


def load_dividends(underlyings_info_folder: str) -> pd.DataFrame:
    """
    :param underlyings_info_folder, the path to dividends, earnings, stock_price_history data

    Load and clean dividends data
    """
    dividends = pd.read_csv(os.path.join(underlyings_info_folder, DIVIDENDS))

    dividends.index = dividends.exDate
    dividends.index.names = ["index"]
    dividends = dividends.drop(columns=["divFreq"])
    dividends = dividends.rename(columns={"divAmt": "dividend_amount"})
    logger.info(f"Loaded Dividends")
    dividends = dividends[dividends.exDate != "2020-01-01"]
    return dividends


def add_realized_mean_returns_and_std(
    G: pd.DataFrame, price_history: pd.DataFrame
) -> pd.DataFrame:
    """
    :param G, the grid of data
    """
    returns = np.log(price_history.dividend_adjusted_prices).diff()
    price_history.loc[:, f"median_variance"] = (
        returns.rolling(5)
        .std()
        .rolling(20, min_periods=20)
        .quantile(0.5, interpolation="linear")
    )
    std = price_history.drop(
        columns=["dividend_adjusted_prices", "dividend_amount", "close"]
    )
    G = G.join(std, how="left")
    return G


def classify_atm_otm_itm(G: pd.DataFrame):
    """
    :param options, options Dataframe
    :G, the grid
    """
    G["sigma_for_moneyness"] = G["median_variance"] * np.sqrt(
        G["business_days_to_maturity"]
    )
    G["logdiff"] = np.log(G["strike"]) - np.log(G["underlying"])
    G["in_moneyness"] = np.nan
    # atm
    indicator = (G["logdiff"] >= -G["sigma_for_moneyness"]) & (
        G["logdiff"] <= G["sigma_for_moneyness"]
    )
    G.loc[indicator, "in_moneyness"] = 0
    # in the money call, out the money put
    indicator = (G["logdiff"] >= -2 * G["sigma_for_moneyness"]) & (
        G["logdiff"] < -G["sigma_for_moneyness"]
    )
    G.loc[indicator, "in_moneyness"] = 1 * G.loc[indicator, "option_type"]

    # out the money call, in the money put
    indicator = (G["logdiff"] > G["sigma_for_moneyness"]) & (
        G["logdiff"] <= 2 * G["sigma_for_moneyness"]
    )
    G.loc[indicator, "in_moneyness"] = -1 * G.loc[indicator, "option_type"]

    # deep otm call, deep itm put
    indicator = G["logdiff"] > 2 * G["sigma_for_moneyness"]
    G.loc[indicator, "in_moneyness"] = -2 * G.loc[indicator, "option_type"]

    # deep itm call, deep otm put
    indicator = G["logdiff"] < -2 * G["sigma_for_moneyness"]
    G.loc[indicator, "in_moneyness"] = 2 * G.loc[indicator, "option_type"]

    G = G[G["in_moneyness"].notna()]
    return G


def build_noise_columns_names(
    option_type: str, all_moneyness: list, all_maturity_buckets: list
) -> list:
    """
    When aggregating pandas as created these noise columns full of zeroes, namely

    caskiv, cbidiv when option_type is (put, -1)
    paskiv, pbidiv when option_type is (call, 1)
    """
    noise_columns = []

    if option_type == "put":
        iv_types = ["paskiv", "pbidiv"]
        option_type_code = 1
    else:
        iv_types = ["caskiv", "cbidiv"]
        option_type_code = -1

    for iv_type in iv_types:
        for moneyness in all_moneyness:
            for maturity_bucket in all_maturity_buckets:
                noise_column = f"{iv_type}_moneyness={moneyness}_maturity_bucket={maturity_bucket}_option_type={option_type_code}"
                noise_columns.append(noise_column)
    return noise_columns


def average_iv_on_same_bucket(G: pd.DataFrame) -> pd.DataFrame:
    """
    We fill iv nans with the mean in same moneyness
    """
    start = time.monotonic()
    logging.info(f"filling na in empty buckets")
    for moneyness in MONEYNESS:
        iv_same_moneyness = [x for x in G.columns if f"iv_moneyness={moneyness}" in x]
        mean_same_moneysness = G[iv_same_moneyness].mean(axis=1)
        for column in iv_same_moneyness:
            G[column] = G[column].fillna(mean_same_moneysness)
    end = time.monotonic()
    logging.info(f"filling na took {end-start} s")
    return G


def reshape_multi_index_to_columns_bucket(grouped: pd.DataFrame) -> pd.DataFrame:
    aggregated = pd.pivot_table(
        grouped,
        index=["trade_date"],
        columns=["in_moneyness", "maturity_bucket", "option_type"],
        values=["volume", "open_interest", "pbidiv", "paskiv", "cbidiv", "caskiv"],
    )
    aggregated.columns = [
        "{}_moneyness={}_maturity_bucket={}_option_type={}".format(j, k, i, z)
        for j, k, i, z in aggregated.columns
    ]
    return aggregated


def drop_noise_columns(aggregated: pd.DataFrame, G: pd.DataFrame) -> pd.DataFrame:
    # The pivoting does NxM, e.g. computes call ask iv when option type is put even if the resulting columns is just NaN and zeroes
    all_maturity_buckets = G.maturity_bucket.unique().tolist()
    all_moneyness = G.in_moneyness.unique().tolist()

    aggregated = aggregated.drop(
        columns=build_noise_columns_names("put", all_moneyness, all_maturity_buckets),
        errors="ignore",
    )
    aggregated = aggregated.drop(
        columns=build_noise_columns_names("call", all_moneyness, all_maturity_buckets),
        errors="ignore",
    )
    return aggregated


# -- Data Description Function! --
def get_number_of_options_per_group(G: pd.DataFrame) -> pd.DataFrame:

    # how many options in each group?
    df = G.groupby(GROUP_DISCRIMINATORS).agg("count")
    df = reshape_multi_index_to_columns_bucket(df)
    df = df.fillna(0)
    return df


def get_number_of_nans_per_column(
    G: pd.DataFrame, data_preprocessor: DataPreprocessor
) -> pd.DataFrame:
    predictors = get_predictors(G)
    nans = pd.DataFrame(pd.isna(G[predictors]).sum(), columns=["number_of_nans"]).T

    max_nans = G.shape[0]
    all_columns = (
        data_preprocessor.iv_columns
        + data_preprocessor.volume_columns
        + data_preprocessor.open_interest_columns
    )

    missing_columns = list(set(all_columns) - set(nans.columns.tolist()))

    # XXX if they get the max punishe
    if len(missing_columns) > 0:
        nans[missing_columns] = max_nans

    return nans


def build_average_columns(G: pd.DataFrame) -> pd.DataFrame:

    start_time = time.monotonic()
    logging.info(
        f"start aggregating per trade_date, in_moneyness, option_type, maturity_bucket"
    )

    # average iv
    grouped = G.groupby(GROUP_DISCRIMINATORS).mean()
    grouped = grouped.drop(columns=["volume", "open_interest"])

    # sum volume, open interest
    grouped_vol_oi = G.groupby(GROUP_DISCRIMINATORS).sum()
    grouped_vol_oi = grouped_vol_oi.drop(
        columns=["caskiv", "cbidiv", "paskiv", "pbidiv"]
    )

    grouped = grouped.join(grouped_vol_oi)
    end_time = time.monotonic()
    logging.info(f"aggregation is complete, it took: {end_time - start_time} s")

    # creating the 280 columns
    start_time = time.monotonic()
    logging.info(f"reshaping dataframe in single line")
    aggregated = reshape_multi_index_to_columns_bucket(grouped)

    aggregated = drop_noise_columns(aggregated, G)

    end_time = time.monotonic()
    logging.info(f"end of reshape {end_time-start_time}")
    return aggregated.copy()


def days_difference(
    G: pd.DataFrame, dates: pd.DataFrame, column_name: str, ticker: str
):
    """
    Compute the number of calendar days between two days
    """
    start_date = G.index.min()
    dates = dates[dates.index >= start_date]

    if dates.empty and column_name == "number_days_to_earnings":
        append_to_file(
            f"{LOG_FOLDER}tickers_with_no_earnings_matching_option_data.log", [ticker]
        )
        logging.warning(f"Ticker={ticker} has no earnings in the option dataset")
        return pd.DataFrame()
    if dates.empty and column_name == "number_days_to_dividends":
        return G

    for date_ in dates.index.tolist():
        idx = (G.index <= date_) & (G.index > start_date)
        G.loc[idx, "closest_date"] = date_
        start_date = date_
    G.closest_date = pd.to_datetime(G.closest_date)
    G[column_name] = G.closest_date - G.trade_date
    G = G.drop(columns=["closest_date"])
    G[column_name] = G[column_name].dt.days
    return G


def add_returns(G: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    return_column_name = "return_t={}"
    for return_days in RETURNS_FOR_STATISTICS:
        prices_ = np.log(prices).diff(return_days).shift(-return_days)

        G[return_column_name.format(return_days)] = prices_

        ## return at time 0
        if return_days == 1:
            prices_ = np.log(prices).diff(return_days)
            G[return_column_name.format(0)] = prices_
    return G


def get_predictors(G: pd.DataFrame) -> list:
    returns = [x for x in G.columns if "return" in x]
    avg_iv = [x for x in G.columns if "expiration_days" in x]
    open_interests = [x for x in G.columns if "open_interest" in x]
    volume = [x for x in G.columns if "volume" in x]
    predictors = G.columns.tolist()
    predictors = set(predictors) - set(DO_NOT_NORMALIZE)
    predictors = set(predictors) - set(avg_iv)
    predictors = set(predictors) - set(open_interests)
    predictors = set(predictors) - set(volume)
    predictors = list(set(predictors) - set(returns))
    return predictors


def build_orats_iv_columns_groups():
    """
    Description
    -------------
    builds the buckets group per type (call/put), (lower/upper, atm)
    and ignores maturities. Used to compute the average

    Input
    -------------
    Void

    Output
    ---------------
    list of names
    """
    iv_groups = dict()
    for iv_type, option_type in [
        ("caskiv", "1"),
        ("cbidiv", "1"),
        ("paskiv", "-1"),
        ("pbidiv", "-1"),
    ]:
        for moneyness in MONEYNESS:
            iv_group = []
            for maturity_bucket in MATURITY_BUCKETS:
                iv_group.append(
                    f"{iv_type}_moneyness={moneyness}_maturity_bucket={maturity_bucket}_option_type={option_type}"
                )
            iv_groups[f"{iv_type}_{moneyness}"] = iv_group
    return iv_groups


def fill_na_with_average_across_buckets(G: pd.DataFrame) -> pd.DataFrame:
    """
    Description
    -----------------
    fill na with average across buckets

    except: DO_NOT_NORMALIZE columns
    and 'return' columns
    Input
    ------------------
    G (pd.DataFrame), the dataframe

    Output
    -------------------
    the filled dataframes
    """
    columns_to_interpolate = get_predictors(G)

    start = time.time()
    iv_columns_group = build_orats_iv_columns_groups()

    for iv_column_group in iv_columns_group.keys():
        columns = iv_columns_group[iv_column_group]
        subset = [x for x in columns if x in G.columns]
        if len(columns) != len(subset):
            iv_columns_group[iv_column_group] = subset
            columns = subset
        G[f"{iv_column_group}_average"] = G[columns].mean(axis=1)

    for date_ in G.index.tolist():
        for iv_column_group in iv_columns_group.keys():
            columns = iv_columns_group[iv_column_group]
            value = G.loc[date_, f"{iv_column_group}_average"]
            G.loc[date_, columns] = G.loc[date_, columns].fillna(value)
    end = time.time()

    logging.info(f"Averaging took {round(end-start, 2)}s")
    avg_exact_days = [x for x in G.columns if "expiration_days" in x]
    columns_to_interpolate = set(columns_to_interpolate) - set(avg_exact_days)
    return G


def remove_outliers(aggregated: pd.DataFrame, return_columns: list) -> pd.DataFrame:
    for return_column in return_columns:
        if return_column != "return_t=252":
            aggregated = aggregated[
                (aggregated[return_column] > -1) & (aggregated[return_column] < 1)
            ]
    return aggregated


def adjust_price_history_from_dividend(
    ticker_price_history: pd.DataFrame, dividends: pd.DataFrame
) -> pd.DataFrame:

    ticker_price_history = ticker_price_history.join(dividends, how="left")
    ticker_price_history.dividend_amount = ticker_price_history.dividend_amount.fillna(
        value=0
    )
    # adjust underlying price
    ticker_price_history["dividend_adjusted_prices"] = (
        ticker_price_history.close + ticker_price_history.dividend_amount
    )
    ticker_price_history = ticker_price_history.drop(columns=["ticker"])
    return ticker_price_history


def create_grid_per_single_stock(
    ticker: str,
    G: pd.DataFrame,
    earnings: pd.DataFrame,
    dividends: pd.DataFrame,
    stock_price_history: pd.DataFrame,
    data_preprocessor: DataPreprocessor,
    save_folder: str,
) -> pd.DataFrame:
    """
    Single stock G
    """
    ticker_price_history = stock_price_history.copy()
    ticker_price_history = adjust_price_history_from_dividend(
        ticker_price_history, dividends
    )

    G.index = G.trade_date

    G = add_realized_mean_returns_and_std(G, ticker_price_history)
    G = classify_atm_otm_itm(G)

    ## for generalized recovery
    atm = G[G.in_moneyness == 0].copy()
    calls = get_iv_at_exact_maturities(atm, option_type=1)
    puts = get_iv_at_exact_maturities(atm, option_type=-1)

    atm = calls.join(puts)

    # time to aggregate (maturity_bucket, in_moneyness)
    # drop useless columns
    sigma_for_moneyness = G.sigma_for_moneyness
    sigma_for_moneyness.index.names = ["index"]
    sigma_for_moneyness = sigma_for_moneyness.groupby("index").first()

    G = G.drop(
        columns=[
            "sigma_for_moneyness",
            "median_variance",
            "bid",
            "ask",
            "underlying",
            "logdiff",
            "number_days",
            "ticker",
            "strike",
            "expiration_date",
            "business_days_to_maturity",
        ]
    )
    G.index.names = ["index"]

    # -- BEGIN:    Data Description --
    number_of_options_per_group = get_number_of_options_per_group(G)
    number_of_options_per_group = drop_noise_columns(number_of_options_per_group, G)
    number_of_options_per_group["ticker"] = ticker
    # -- END:      Data Description --
    G = build_average_columns(G)

    # -- BEGIN:    Data Description --
    nans = get_number_of_nans_per_column(G, data_preprocessor)
    nans["ticker"] = ticker
    os.makedirs(f"{save_folder}/nans_statistics/", exist_ok=True)
    nans.to_pickle(f"{save_folder}/nans_statistics/{ticker}.p", protocol=3)
    # -- END:      Data Description --

    # --- BEGIN:    Dealing with NaN's --
    # volume, open interest -> set to zero
    volume_open_interest_columns = [
        x for x in G.columns.tolist() if "volume" in x or "open_interest" in x
    ]
    G[volume_open_interest_columns] = G[volume_open_interest_columns].fillna(value=0)

    # iv -> average same maturity same moneyness, reason: smile
    G = average_iv_on_same_bucket(G)
    # -- END:         Dealing with Nan's--

    G["sigma_for_moneyness"] = sigma_for_moneyness
    # add number to earnings date
    G["trade_date"] = pd.to_datetime(G.index)

    G = days_difference(G, earnings, "number_days_to_earnings", ticker)
    # ticker has no earnings date in the option data dataset
    if G.empty:
        return G

    if not dividends.empty:
        G = days_difference(G, dividends, "number_days_to_dividends", ticker)
    try:
        G = G.drop(columns=["trade_date"])
    except:
        raise Exception(f"{ticker}")
    # we predict X business days ahead
    prices = ticker_price_history.dividend_adjusted_prices
    G = add_returns(G, prices)
    G = remove_outliers(G, data_preprocessor.returns_columns)

    G = G.dropna(how="all", subset=["number_days_to_earnings"])
    # G = G.dropna(how='all', subset=['number_days_to_dividends'])

    ticker_price_history = ticker_price_history.drop(columns=["median_variance"])
    G = G.join(ticker_price_history)
    G = G.join(earnings, how="left")
    G = G.join(atm, how="left")
    G.earnings_flag = G.earnings_flag.fillna(value=0)

    number_of_options_per_group = number_of_options_per_group.loc[G.index]
    number_of_options_per_group["trade_date"] = number_of_options_per_group.index
    os.makedirs(f"{save_folder}/number_of_options_per_group/", exist_ok=True)
    number_of_options_per_group.to_pickle(
        f"{save_folder}/number_of_options_per_group/{ticker}.p", protocol=3
    )
    return G


def add_historical_rolling_volume_open_interest(
    G: pd.DataFrame, data_preprocessor: DataPreprocessor
) -> pd.DataFrame:
    """
    Compute average volume/open_interest on rolling 20 days
    Compute current volume/open_interest / rolling 252 days mean
    """
    # XXX some stocks don't have very long maturity options e.g. >250
    volume_columns = [x for x in data_preprocessor.volume_columns if x in G.columns]
    open_interest_colums = [
        x for x in data_preprocessor.open_interest_columns if x in G.columns
    ]

    rolling_volume_columns = [
        x.replace("volume", "rolling_20_volume") for x in volume_columns
    ]
    rolling_open_interest_columns = [
        x.replace("open_interest", "rolling_20_open_interest")
        for x in open_interest_colums
    ]

    rolling_volume_mean_per_year_columns = [
        x.replace("volume", "rolling_mean_volume_per_year") for x in volume_columns
    ]
    rolling_open_interest_mean_per_year_columns = [
        x.replace("open_interest", "rolling_mean_open_interest_per_year")
        for x in open_interest_colums
    ]

    # XXX Rolling average
    G[rolling_volume_columns] = G[volume_columns].rolling(20).mean()
    G[rolling_open_interest_columns] = G[open_interest_colums].rolling(20).mean()

    # XXX value/rolling average 252 days
    G[rolling_volume_mean_per_year_columns] = (
        G[volume_columns] / G[volume_columns].rolling(252).mean()
    )

    G[rolling_open_interest_mean_per_year_columns] = (
        G[open_interest_colums] / G[open_interest_colums].rolling(252).mean()
    )
    return G


def trim_data(G: pd.DataFrame) -> pd.DataFrame:
    """
    Trim the quantiles 0.005 and 0.995
    """
    start_time = time.time()
    predictors = get_predictors(G)
    for predictor in predictors:
        quantile_005 = G[predictor].quantile(0.005)
        quantile_995 = G[predictor].quantile(0.995)

        G[predictor] = [quantile_005 if x < quantile_005 else x for x in G[predictor]]
        G[predictor] = [quantile_995 if x > quantile_995 else x for x in G[predictor]]

    end_time = time.time()
    logging.info(f"trimming data took: {end_time - start_time}s")
    return G


def get_iv_at_exact_maturities(
    options_raw: pd.DataFrame, option_type: int
) -> pd.DataFrame:
    """

    Compute the average iv between c/paskiv and c/pbidiv
    and compute the average iv per expiration_days in [1,5,20,252]
    """
    options = options_raw[
        (options_raw.business_days_to_maturity == 1)
        | (options_raw.business_days_to_maturity == 5)
        | (options_raw.business_days_to_maturity == 20)
        | (options_raw.business_days_to_maturity == 252)
    ]

    options = options[options.option_type == option_type].copy()

    if option_type == 1:
        new_column = "cavgiv"
        options[new_column] = (options["caskiv"] + options["cbidiv"]) / 2
    else:
        new_column = "pavgiv"
        options[new_column] = (options["paskiv"] + options["pbidiv"]) / 2

    options = options[["business_days_to_maturity", "trade_date", new_column]]
    options = options.groupby(["trade_date", "business_days_to_maturity"]).mean()
    options = pd.pivot_table(
        options,
        index=["trade_date"],
        columns=["business_days_to_maturity"],
        values=[new_column],
    )
    options.columns = ["{}_expiration_days={}".format(j, k) for j, k in options.columns]

    options = options.sort_index()
    return options


def add_business_days_to_maturity(
    options: pd.DataFrame, business_days: pd.DataFrame
) -> pd.DataFrame:

    options = options[options.trade_date <= "2020-08-31"]
    options.index = options[["trade_date", "expiration_date"]]
    business_days.index = business_days[["trade_date", "expirdate"]]
    business_days = business_days.drop(columns=["trade_date", "expirdate"])
    options = options.join(business_days, how="left")
    options = options.reset_index(drop=True)
    options_maturity = options[~options.business_days_to_maturity.isna()].copy()
    try:
        assert pd.isna(options_maturity.business_days_to_maturity).mean() == 0
        output = options_maturity
    except:

        business_days = options[["trade_date", "expiration_date"]].drop_duplicates()

        business_days["business_days_to_maturity"] = business_days.apply(
            lambda x: business_days_between(x.trade_date, x.expiration_date), axis=1
        )
        options.pop("business_days_to_maturity")
        options.index = options[["trade_date", "expiration_date"]]
        business_days.index = business_days[["trade_date", "expiration_date"]]
        business_days = business_days.drop(columns=["trade_date", "expiration_date"])
        options = options.join(business_days, how="left")
        options = options.reset_index(drop=True)
        assert pd.isna(options.business_days_to_maturity).mean() == 0
        output = options
    return output


def compute_grid(
    option_data: str,
    save_folder: str,
    data_folder: str,
    dividends: pd.DataFrame,
    earnings: pd.DataFrame,
    stocks_price_history: pd.DataFrame,
    business_days: pd.DataFrame,
    data_description_only: bool,
):
    """
    Load single ticker data
    """
    start_time = time.monotonic()

    ticker = option_data.replace(".csv", "")
    logging.info("-" * 47)
    logging.info("Processing: {:<15}".format(ticker))
    logging.info("-" * 47)
    # Object in charge to preprocess data
    data_preprocessor = DataPreprocessor()

    # options data
    ticker_options = load_option_data(data_folder, option_data)
    ticker_options = add_business_days_to_maturity(ticker_options, business_days)
    ticker_options = add_maturity_bucket(ticker_options)
    start_date = ticker_options.trade_date.min()

    logging.info(f"Option Data for ticker={ticker} start_date={start_date}")

    # earnings
    ticker_earnings = earnings[
        (earnings.ticker == ticker) & (earnings.earnDate > start_date)
    ].copy()
    ticker_earnings = ticker_earnings.drop(columns=["ticker", "earnDate"])

    # dividends
    ticker_dividends = dividends[
        (dividends.ticker == ticker) & (dividends.exDate > start_date)
    ].copy()
    ticker_dividends = ticker_dividends.drop(columns=["ticker", "declaredDate"])

    ticker_stock_price_history = stocks_price_history[
        (stocks_price_history.ticker == ticker)
        & (stocks_price_history.time > start_date)
    ]

    ticker_dividends = data_preprocessor.drop_duplicates(
        ticker_dividends, unique_column="exDate"
    )
    ticker_stock_price_history = data_preprocessor.drop_duplicates(
        ticker_stock_price_history, unique_column="time"
    )
    ticker_earnings = data_preprocessor.drop_duplicates(
        ticker_earnings, unique_column="index"
    )

    ## XXX pre-sort everything
    ticker_dividends = ticker_dividends.sort_index()
    ticker_earnings = ticker_earnings.sort_index()
    ticker_stock_price_history = ticker_stock_price_history.sort_index()
    # create 281 columns
    G = create_grid_per_single_stock(
        ticker,
        ticker_options,
        ticker_earnings,
        ticker_dividends,
        ticker_stock_price_history,
        data_preprocessor,
        save_folder,
    )
    end_time = time.monotonic()
    logging.info(
        f"Processing time to build grid G for ticker={ticker}: {end_time-start_time}s"
    )

    if not data_description_only:

        if G.empty:
            return
        G["ticker"] = ticker

        G = fill_na_with_average_across_buckets(G)

        if G.empty:
            return
        G = trim_data(G)

        G = add_historical_rolling_volume_open_interest(G, data_preprocessor)
        assert G.shape[0] == G.index.unique().shape[0]
        G.to_csv(f"{save_folder}{option_data}")


def create_grid_of_maturity_and_strike(
    data_folder: str,
    underlyings_info_folder: str,
    business_days_file: str,
    recompute: bool,
    debug: bool,
    data_description_only: bool,
    remove_tickers: list,
    subset_tickers: list,
):
    """
    :param data_folder, the path to option data
    :param underlyings_info_folder, the path to dividends, earnings, stock_price_history data

    Big G for all stocks

    G is dimension of flatten grid of maturity and strike (#bins_strike*#bins_maturity).For every stcok, and every day I have
        - a grid of dim G -> x
        - stock price at time t+1->yWhat I feed my nnet:
    """
    # load dividends
    dividends = load_dividends(underlyings_info_folder)

    # load earnings
    earnings = load_earnings(underlyings_info_folder)

    # load stock prices
    stocks_price_history = pd.read_csv(
        os.path.join(underlyings_info_folder, STOCK_PRICES)
    )
    logger.info(f"Loaded Stocks Price History")
    # load buisiness days
    business_days = pd.read_csv(business_days_file)

    # options_data
    options_data = [x for x in os.listdir(data_folder) if ".csv" in x]

    if subset_tickers:
        options_data = [f"{x}.csv" for x in subset_tickers]

    tickers_in_history = stocks_price_history.ticker.unique().tolist()
    options_data = [
        x for x in options_data if x.replace(".csv", "") in tickers_in_history
    ]

    # we care about tickers with earnings only!
    # That's because calendar days to earnings matter as a predictor
    tickers_with_earnings = earnings.ticker.unique().tolist()
    options_data = [
        x for x in options_data if x.replace(".csv", "") in tickers_with_earnings
    ]

    # useful for debug purpose
    if remove_tickers:
        options_data = [
            x for x in options_data if not x.replace(".csv", "") in remove_tickers
        ]

    save = "/".join(data_folder.split("/")[:-2])
    save_folder = f"{save}/grids/"

    os.makedirs(save_folder, exist_ok=True)

    if not recompute:
        already_computed = [x for x in os.listdir(save_folder)]
        options_data = list(set(options_data) - set(already_computed))

    options_data.sort()
    logging.info(f"Total Options: {len(options_data)}")
    if not debug:
        pool = mp.Pool(50)
        pool.starmap(
            compute_grid,
            [
                (
                    option_data,
                    save_folder,
                    data_folder,
                    dividends,
                    earnings,
                    stocks_price_history,
                    business_days,
                    data_description_only,
                )
                for option_data in options_data
            ],
        )

        pool.close()
        pool.join()

    else:
        for option_data in options_data:
            compute_grid(
                option_data,
                save_folder,
                data_folder,
                dividends,
                earnings,
                stocks_price_history,
                business_days,
                data_description_only,
            )


if __name__ == "__main__":
    """
    This file computes the grids of predictors
    for each ticker
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--option-data-folder",
        dest="option_data_folder",
        type=str,
        required=True,
        help="Path to the folder which contains all raw option data",
    )

    parser.add_argument(
        "--underlyings-info-folder",
        dest="underlyings_info_folder",
        type=str,
        required=True,
        help="Path to the folder which contains all additional underlying info",
    )

    parser.add_argument(
        "--business-days-file",
        dest="business_days_file",
        type=str,
        required=True,
        help="path to the business days folder",
    )

    parser.add_argument(
        "--recompute",
        dest="recompute",
        action="store_true",
        help="recompute will ignore already computed files",
    )
    parser.set_defaults(recompute=False)

    parser.add_argument(
        "--data-description-only",
        dest="data_description_only",
        action="store_true",
        help="compute only the data description file",
    )
    parser.set_defaults(data_description_only=False)

    parser.add_argument(
        "--remove-tickers",
        dest="remove_tickers",
        type=str,
        help="path to file with tickers with too many gaps",
    )

    parser.add_argument(
        "--subset-tickers",
        dest="subset_tickers",
        type=str,
        help="path to file with a subset of tickers to re-compute",
    )

    parser.add_argument("--debug", dest="debug", action="store_true", help="DEBUG")
    parser.set_defaults(debug=False)

    args = parser.parse_args()
    remove_tickers = []

    # in case we want to remove some tickers
    if args.remove_tickers:
        df = pd.read_csv(args.remove_tickers, names=["ticker"])
        remove_tickers = df.ticker.tolist()
    subset_tickers = []

    # in case we want to recompute just a grid subset
    if args.subset_tickers:
        df = pd.read_csv(args.subset_tickers, names=["ticker"])
        subset_tickers = df.ticker.tolist()

    create_grid_of_maturity_and_strike(
        args.option_data_folder,
        args.underlyings_info_folder,
        args.business_days_file,
        args.recompute,
        args.debug,
        args.data_description_only,
        remove_tickers,
        subset_tickers,
    )
