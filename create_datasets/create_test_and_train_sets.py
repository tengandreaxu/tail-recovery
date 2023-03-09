import os
import argparse
import pandas as pd
import numpy as np
from create_datasets.constants import DATASETS_DIRECTORIES
from create_datasets.DataPreprocessor import DataPreprocessor

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("testTrainCreator")
pd.set_option("display.max_colwidth", 300)


def create_directory(directory: str):
    try:
        os.mkdir(directory)
        logger.info(f"folder={directory} created")
    except:
        logger.warning(f"folder={directory} already exists")


def get_volume_columns(df: pd.DataFrame) -> list:
    volume_columns = [x for x in df.columns if "volume_" in x]
    return volume_columns


def get_iv_columns(df: pd.DataFrame) -> list:
    ivs = [x for x in df.columns if "iv_" in x]
    avgs = [x for x in df.columns if "avgiv" in x]
    output = list(set(ivs) - set(avgs))
    return output


def get_open_interest_columns(df: pd.DataFrame) -> list:
    oi = [x for x in df.columns if "open_interest_" in x]
    return oi


def get_feature_type(feature_name: str):
    if "iv_" in feature_name:
        return "ivs"
    elif "open_interest_" in feature_name:
        return "open_interests"
    elif "volume_" in feature_name:
        return "volumes"
    else:
        return "other"


def save_trade_date_ticker(df: pd.DataFrame, data_folder: str, file_name: str):
    file_ = os.path.join(data_folder, file_name)
    df.to_csv(file_)
    logging.info(f"saved: {file_}")


def create_all_directories_per_dataset(x: str, add_lags: bool):
    create_directory(x)

    for sub_directory in DATASETS_DIRECTORIES:
        file_ = os.path.join(x, sub_directory)
        create_directory(file_)
        if add_lags:
            for lag in [-1, -2]:
                create_directory(f"{file_}_t={lag}")


def save_all(
    df: pd.DataFrame,
    return_columns: list,
    iv_columns: list,
    oi_columns: list,
    volume_columns: list,
    tickers_csv: pd.DataFrame,
    x_name: str,
    ticker_name: str,
    add_lags: bool,
    data_preprocessor: DataPreprocessor,
):
    x = df.drop(columns=return_columns)
    subsets = [
        iv_columns,
        oi_columns,
        volume_columns,
        ["number_days_to_earnings", "number_days_to_dividends", "sigma_for_moneyness"],
    ]

    for subset in subsets:

        x_subset_name = "/".join(x_name.split("/")[:-1])
        file_name = x_name.split("/")[-1]
        logging.info(f"Saving {x_subset_name}")
        feature_type = get_feature_type(subset[0])
        if feature_type == "ivs":
            subset = [x for x in subset if "average" not in x]
        x_subset = x[subset].copy()
        x = x.drop(columns=subset)

        output_dir = os.path.join(x_subset_name, feature_type)

        x_subset = normalize(x_subset, data_preprocessor.returns_columns, output_dir)

        x_subset.to_pickle(os.path.join(output_dir, f"{file_name}.p"), protocol=4)
        if add_lags:
            for lag in [-1, -2]:
                x_subset.shift(-1 * lag).to_pickle(
                    f"{x_subset_name}/{feature_type}_t={lag}/{file_name}.p", protocol=3
                )

    tickers_csv.to_pickle(ticker_name, protocol=4)


def create_train_data(
    aggregated: pd.DataFrame,
    data_folder: str,
    return_columns: list,
    iv_columns: list,
    oi_columns: list,
    volume_columns: list,
    ticker_csv: pd.DataFrame,
    add_lags: bool,
    data_preprocessor: DataPreprocessor,
):
    X_train_folder = os.path.join(data_folder, "X_train")

    create_all_directories_per_dataset(X_train_folder, add_lags)

    save_all(
        df=aggregated,
        return_columns=return_columns,
        iv_columns=iv_columns,
        oi_columns=oi_columns,
        volume_columns=volume_columns,
        tickers_csv=ticker_csv,
        x_name=os.path.join(X_train_folder, "X_train"),
        ticker_name=os.path.join(X_train_folder, "train_ticker.p"),
        add_lags=add_lags,
        data_preprocessor=data_preprocessor,
    )


def normalize(
    aggregated: pd.DataFrame, return_columns: str, data_folder: str
) -> pd.DataFrame:
    """
    normalize columns
    """
    logging.info(f"start normalizing")
    all_columns = aggregated.columns.tolist()

    to_remove = [
        "number_days_to_earnings",
        "number_days_to_dividends",
        "ticker",
        "close",
        "dividend_adjusted_prices",
        "dividend_amount",
        "earnings_flag",
    ]

    to_remove += return_columns
    for column in to_remove:
        if column in all_columns:
            all_columns.remove(column)

    means = aggregated[all_columns].mean()
    stds = aggregated[all_columns].std()
    means.to_pickle(os.path.join(data_folder, "means.p"))
    stds.to_pickle(os.path.join(data_folder, "stds.p"))

    aggregated[all_columns] = (aggregated[all_columns] - means) / stds
    return aggregated


def add_realized_historical_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds realized/historical volatility
    """
    for rolling_window in [5, 20, 252]:
        df[f"rolling_historical_std_window={rolling_window}"] = (
            df["return_t=0"].rolling(rolling_window).std()
        )

        ## Compute rolling windows as normal then shift in the past to make it future
        df[f"rolling_realized_std_window={rolling_window}"] = (
            df["return_t=0"].rolling(rolling_window).std().shift(rolling_window)
        )
        df["squared_mean"] = df["return_t=0"] ** 2
        df[f"rolling_historical_std_zero_mean_window={rolling_window}"] = np.sqrt(
            (df["squared_mean"].rolling(rolling_window).sum()) / (rolling_window - 1)
        )
        df[f"rolling_realized_std_zero_mean_window={rolling_window}"] = np.sqrt(
            (df["squared_mean"].rolling(rolling_window).sum()) / (rolling_window - 1)
        ).shift((rolling_window))
    return df


def create_aggregated(
    all_grids: list,
    data_folder: str,
    load_from_checkpoint: bool,
) -> pd.DataFrame:
    aggregated = pd.DataFrame()
    totals = len(all_grids)
    count = 0
    temp_folder = os.path.join(data_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)

    if load_from_checkpoint:
        last_checkpoint = os.listdir(temp_folder)
        last_checkpoint.sort()
        last_checkpoint = last_checkpoint[-1]

        number_of_last_ticker = int(
            last_checkpoint.replace("aggregated", "").replace(".p", "")
        )
        aggregated = pd.read_pickle(os.path.join(temp_folder, last_checkpoint))
        all_grids = all_grids[number_of_last_ticker:]
    for grid in all_grids:
        logger.info(f"reading: {grid}")
        df = pd.read_csv(f"{data_folder}/{grid}")

        ## XXX Per ticker rolling window standard deviation
        df = add_realized_historical_volatility(df)

        columns = df.columns.tolist()
        columns.remove("trade_date")
        columns.remove("ticker")

        df[columns] = df[columns].astype(np.float32)
        aggregated = aggregated.append(df)
        logging.info(f"aggregated shape: {aggregated.shape}")
        totals -= 1
        logging.info(f"files remaining: {totals}")
        logging.info(f"{aggregated.info()}")

        count += 1
        if count % 1000 == 0:
            # checkpoint
            aggregated.to_pickle(os.path.join(temp_folder, f"aggregate{count}.p"))

    aggregated.to_pickle(os.path.join(data_folder, "aggregate.p"))

    return aggregated


def elaborate_aggregate(aggregated: pd.DataFrame, data_folder: str, add_lags: bool):

    """
    From an aggregate create train dataset
    """

    aggregated.index = aggregated.trade_date
    aggregated = aggregated.drop(columns=["trade_date"])
    data_preprocessor = DataPreprocessor()

    return_columns = data_preprocessor.returns_columns
    ## XXX ATM unstardized call put
    atm_columns = data_preprocessor.atm_iv_columns
    iv_exact_maturity = [x for x in aggregated.columns if "expiration_days" in x]
    historical_realized_std = [
        x
        for x in aggregated.columns
        if "rolling_realized" in x or "rolling_historical" in x
    ]
    ticker_csv_columns = (
        atm_columns
        + return_columns
        + iv_exact_maturity
        + [
            "ticker",
            "close",
            "dividend_adjusted_prices",
            "dividend_amount",
            "number_days_to_earnings",
        ]
        + historical_realized_std
    )

    ticker_csv = aggregated[ticker_csv_columns].copy()
    aggregated = aggregated.drop(columns=historical_realized_std)
    assert aggregated.shape[0] == ticker_csv.shape[0]

    # columns
    iv_columns = get_iv_columns(aggregated)
    oi_columns = get_open_interest_columns(aggregated)
    volume_columns = get_volume_columns(aggregated)

    # aggregated = aggregated.dropna(how='all', subset=[predicted_variable])
    assert aggregated.shape[0] == ticker_csv.shape[0]
    aggregated.index = pd.to_datetime(aggregated.index)

    create_train_data(
        aggregated,
        data_folder,
        return_columns,
        iv_columns,
        oi_columns,
        volume_columns,
        ticker_csv,
        add_lags,
        data_preprocessor,
    )


def create_test_and_training_sets(
    data_folder: str, save_aggregate: bool, add_lags: bool, load_from_checkpoint: bool
):
    """
    :param data_folder, path to the folder containing the grids
    """

    all_grids = [x for x in os.listdir(data_folder) if ".csv" in x]

    ## XXX Append sorted
    all_grids.sort()
    if not save_aggregate:
        logging.info(f"Loading the previous saved aggregate")
        aggregated = pd.read_pickle(os.path.join(data_folder, "aggregate.p"))
    else:
        aggregated = create_aggregated(all_grids, data_folder, load_from_checkpoint)
    elaborate_aggregate(aggregated, data_folder, add_lags)


if __name__ == "__main__":
    """
    This function:

    1. reads all grids created using create_grids.py
    2. counts NaN statistics per trade_date, stock
    3. aggregate in a single dataframe all data
    4. takes out 2020 data -> test_future
    5. takes out 5% -> test_past
    6. the data left -> tain
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-folder",
        dest="data_folder",
        type=str,
        help="path to the folder containing all grids",
        required=True,
    )

    parser.add_argument(
        "--save-aggregate",
        dest="save_aggregate",
        action="store_true",
        help="either or not save the aggregate",
    )
    parser.set_defaults(save_aggregate=False)

    parser.add_argument(
        "--add-lags",
        dest="add_lags",
        action="store_true",
        help="either or not adding lags",
    )
    parser.set_defaults(add_lags=False)

    parser.add_argument(
        "--load-from-checkpoint",
        dest="load_from_checkpoint",
        action="store_true",
        help="either load from checkpoint or not",
    )
    parser.set_defaults(add_lags=False)

    args = parser.parse_args()

    create_test_and_training_sets(
        args.data_folder, args.save_aggregate, args.add_lags, args.load_from_checkpoint
    )
