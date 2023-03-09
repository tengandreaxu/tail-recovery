import argparse
import os
import time
import pandas as pd
import multiprocessing as mp
from plotting.Plotter import Plotter
from create_datasets.create_grids import (
    classify_atm_otm_itm,
    add_business_days_to_maturity,
    add_realized_mean_returns_and_std,
    adjust_price_history_from_dividend,
    split_put_call_info,
)
from util.dataframes_handling import create_dataframe_for_smooth_plot
import logging

logging.basicConfig(level=logging.INFO)


def keep_otm_only(
    df: pd.DataFrame,
    stock_price_history: pd.DataFrame,
    business_days: pd.DataFrame,
    dividends: pd.DataFrame,
) -> pd.DataFrame:
    # we don't need it
    df["number_days"] = 0

    df = split_put_call_info(df)
    try:
        df = add_business_days_to_maturity(df, business_days)
        assert df[df.business_days_to_maturity.isna()].shape[0] == 0
    except:
        return pd.DataFrame()
    stock_price_history.index = stock_price_history.time
    stock_price_history.index.names = ["index"]
    stock_price_history.pop("time")
    price_history = adjust_price_history_from_dividend(stock_price_history, dividends)
    df.index = df.trade_date
    df = add_realized_mean_returns_and_std(df, price_history)
    df = classify_atm_otm_itm(df)
    df = df[(df.in_moneyness == -1) | (df.in_moneyness == -2)]
    return df


def create_volume_dataset(
    file_: list,
    volume_output_dir: str,
    otm_only: bool,
    stock_price_history: pd.DataFrame,
    tickers_used: list,
    business_days: pd.DataFrame,
    dividends: pd.DataFrame,
) -> pd.DataFrame:

    start = time.monotonic()
    ticker = file_.split("/")[-1].replace(".csv", "")

    if ticker not in tickers_used:
        return pd.DataFrame()

    df = pd.read_csv(file_)

    if otm_only:
        try:
            dividends_ticker = dividends[dividends.ticker == ticker].copy()
            dividends_ticker.pop("ticker")
            df = keep_otm_only(
                df,
                stock_price_history[stock_price_history.ticker == ticker].copy(),
                business_days,
                dividends_ticker,
            )
            df["trade_date"] = df.index
            df = df.reset_index(drop=True)
            df = (
                df.groupby(["trade_date", "option_type"])[["volume"]]
                .agg("sum")
                .reset_index()
            )

            puts = df[df.option_type == -1].copy()
            calls = df[df.option_type == 1].copy()

            puts = puts.rename(columns={"volume": "pvolu"})
            calls = calls.rename(columns={"volume": "cvolu"})

            puts.index = puts.trade_date
            calls.index = calls.trade_date

            puts = puts.drop(columns=["trade_date", "option_type"])
            calls = calls.drop(columns=["trade_date", "option_type"])

            df = calls.join(puts, how="left")
            df.cvolu = df.cvolu.fillna(value=0)
            df.pvolu = df.pvolu.fillna(value=0)
            df = df.reset_index()
        except:
            return pd.DataFrame()
    df = df[["pvolu", "cvolu", "trade_date"]]
    end = time.monotonic()
    process_time = round(end - start, 2)
    logging.info("{:<35} {:<25}".format(f"Reading: {file_}", f"Time: {process_time}s"))
    return df


def create_big_volume_dataset(
    files_: list,
    volume_output_dir: str,
    otm_only: bool,
    stock_price_history: pd.DataFrame,
    tickers_used: list,
    business_days: pd.DataFrame,
    dividends: pd.DataFrame,
    parallel: bool,
) -> pd.DataFrame:
    """
    Reads all files and gives back the giant volume only dataset
    """
    dataset = pd.DataFrame()
    outputs = []
    if parallel:
        pool = mp.Pool(40)
        outputs += pool.starmap(
            create_volume_dataset,
            [
                (
                    file_,
                    volume_output_dir,
                    otm_only,
                    stock_price_history,
                    tickers_used,
                    business_days,
                    dividends,
                )
                for file_ in files_
            ],
        )
        pool.close()
        pool.join()
    else:
        for file_ in files_:
            outputs.append(
                create_volume_dataset(
                    file_,
                    volume_output_dir,
                    otm_only,
                    stock_price_history,
                    tickers_used,
                    business_days,
                    dividends,
                )
            )
    dataset = pd.concat(outputs)
    dataset = dataset.groupby("trade_date")[["pvolu", "cvolu"]].agg("sum").reset_index()
    dataset.to_pickle(
        os.path.join(volume_output_dir, f"volume_only_dataset_otm_only={otm_only}.p"),
        protocol=4,
    )
    return dataset


if __name__ == "__main__":
    """
    This scripts reads the raw datasets
    and computes daily call and put volumes
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        type=str,
        dest="data_path",
        help="path to folder which contains raw orats data",
    )

    parser.add_argument(
        "--volume-only-file",
        type=str,
        dest="volume_file",
        help="in case we already computed the volume only file we can just load it",
    )

    parser.add_argument(
        "--is-big",
        action="store_true",
        dest="is_big",
        help="either is the small dataset or is the whole",
    )
    parser.set_defaults(is_big=False)

    parser.add_argument(
        "--remove-extreme",
        action="store_true",
        dest="remove_extreme",
        help="probably data glitch remove those after the .99 percentile",
    )

    parser.add_argument(
        "--business-days-file",
        type=str,
        dest="business_days",
        help="path to the business-days",
    )

    parser.add_argument(
        "--tickers-used", type=str, dest="tickers_used", help="path to the tickers used"
    )

    parser.add_argument(
        "--stock-price-history",
        type=str,
        dest="stock_price_history",
        help="path to the stock price history",
    )
    parser.add_argument(
        "--dividends", type=str, dest="dividends", help="path to dividends"
    )

    parser.add_argument(
        "--otm-only",
        action="store_true",
        dest="otm_only",
        help="either compute the total volume using otm only",
    )
    parser.set_defaults(otm_only=False)
    parser.add_argument(
        "--parallel", action="store_true", dest="parallel", help="parallelize"
    )
    parser.set_defaults(parallel=False)

    args = parser.parse_args()

    if not args.data_path and not args.volume_file:
        logging.error(
            "please specify at least one between data-path and volume-only-file"
        )
        exit(1)

    if args.otm_only and (
        not args.stock_price_history or not args.business_days or not args.dividends
    ):
        logging.error("you must specify the stock price history for otm only")
        exit(1)

    stock_price_history = pd.DataFrame()
    if args.stock_price_history:
        stock_price_history = pd.read_csv(args.stock_price_history)

    dividends = pd.DataFrame()
    if args.dividends:
        dividends = pd.read_csv(args.dividends)
        dividends = dividends.rename(columns={"divAmt": "dividend_amount"})
        dividends.index = dividends.exDate
        dividends.index.names = ["index"]
        dividends = dividends.drop(columns=["divFreq", "declaredDate", "exDate"])

    business_days = pd.DataFrame()
    if args.business_days:
        business_days = pd.read_csv(args.business_days)

    tickers_used = []
    if args.tickers_used:
        tickers_used = pd.read_csv(args.tickers_used)
        tickers_used = tickers_used.ticker.tolist()

    prefix_name = "data_sample=small_"
    if args.is_big:
        prefix_name = "data_sample=big_"

    if args.otm_only:
        prefix_name = prefix_name + "otm_only_"

    print(f"Data Path: {args.data_path}")

    if args.volume_file:
        df = pd.read_pickle(args.volume_file)
    else:
        files_ = os.listdir(args.data_path)
        files_ = [os.path.join(args.data_path, x) for x in files_ if ".csv" in x]
        files_.sort()
        volume_output_dir = os.path.join(args.data_path, "volume_only")
        os.makedirs(volume_output_dir, exist_ok=True)
        df = create_big_volume_dataset(
            files_,
            volume_output_dir,
            args.otm_only,
            stock_price_history,
            tickers_used,
            business_days,
            dividends,
            args.parallel,
        )

    df.trade_date = pd.to_datetime(df.trade_date).dt.date
    df = df.groupby("trade_date")["cvolu", "pvolu"].sum()
    df = df.sort_index()

    df = create_dataframe_for_smooth_plot(df, columns=["cvolu", "pvolu"])

    output_dir = "paper/res_paper/data_statistics/volume_count"
    os.makedirs(output_dir, exist_ok=True)

    plotter = Plotter()
    plotter.plot_multiple_lines_given_row_and_columns(
        df=df,
        row="date",
        columns=["cvolu", "pvolu"],
        labels=["Call Volume", "Put Volume"],
        colors=["black", "brown"],
        linestyles=["solid", "dashdot"],
        title="",
        xlabel="Volume",
        ylabel="Year",
        save_name=os.path.join(
            output_dir, f"{prefix_name}call_and_put_volume_over_years.png"
        ),
        format_dates=True,
    )
