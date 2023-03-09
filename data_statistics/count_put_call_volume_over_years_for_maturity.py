import argparse
import os
import time
import pandas as pd
import multiprocessing as mp
from create_datasets.create_grids import (
    add_business_days_to_maturity,
    add_maturity_bucket,
)
from plotting.Plotter import Plotter
from util.dataframes_handling import create_dataframe_for_smooth_plot


from create_datasets.constants import MATURITY_BUCKETS
import logging

logging.basicConfig(level=logging.INFO)

MATURITY_LABELS = [
    "$0 <= T < 5$",
    "$5 <= T < 15$",
    "$15 <= T < 30$",
    "$30 <= T < 60$",
    "$60 <= T < 120$",
    "$120 <= T < 250$",
    "$T > 250$",
]


def create_volume_dataset_per_maturity(
    file_: list, tickers_used: list, business_days: pd.DataFrame
) -> pd.DataFrame:

    start = time.monotonic()
    ticker = file_.split("/")[-1].replace(".csv", "")

    if ticker not in tickers_used:
        return pd.DataFrame()

    df = pd.read_csv(file_)
    df = df.rename(columns={"expirdate": "expiration_date"})
    try:
        df = add_business_days_to_maturity(df, business_days)
    except:
        return pd.DataFrame()
    df = add_maturity_bucket(df)
    df = df[["pvolu", "cvolu", "trade_date", "maturity_bucket"]]
    end = time.monotonic()
    process_time = round(end - start, 2)
    logging.info("{:<35} {:<25}".format(f"Reading: {file_}", f"Time: {process_time}s"))
    return df


def create_big_volume_dataset_per_maturity(
    files_: list,
    volume_output_dir: str,
    tickers_used: list,
    business_days: pd.DataFrame,
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
            create_volume_dataset_per_maturity,
            [(file_, tickers_used, business_days) for file_ in files_],
        )
        pool.close()
        pool.join()
    else:
        for file_ in files_[:10]:
            outputs.append(
                create_volume_dataset_per_maturity(file_, tickers_used, business_days)
            )
    dataset = pd.concat(outputs)
    dataset = (
        dataset.groupby(["trade_date", "maturity_bucket"])[["pvolu", "cvolu"]]
        .agg("sum")
        .reset_index()
    )
    dataset.to_pickle(
        os.path.join(volume_output_dir, f"volume_only_dataset_per_maturity.p"),
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
        "--parallel",
        action="store_true",
        dest="parallel",
        help="either use multiprocessing or not",
    )
    parser.set_defaults(parallel=False)

    parser.add_argument(
        "--business-days-file",
        type=str,
        dest="business_days",
        help="path to the business-days",
    )

    parser.add_argument(
        "--tickers-used", type=str, dest="tickers_used", help="path to the tickers used"
    )

    args = parser.parse_args()

    if not args.data_path and not args.volume_file:
        logging.error(
            "please specify at least one between data-path and volume-only-file"
        )
        exit(1)

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

    prefix_name = prefix_name + "_per_maturity"

    if args.volume_file:
        df = pd.read_pickle(args.volume_file)
    else:
        files_ = os.listdir(args.data_path)
        files_ = [os.path.join(args.data_path, x) for x in files_ if ".csv" in x]
        files_.sort()

        volume_output_dir = os.path.join(args.data_path, "volume_only")
        os.makedirs(volume_output_dir, exist_ok=True)
        df = create_big_volume_dataset_per_maturity(
            files_, volume_output_dir, tickers_used, business_days, args.parallel
        )
    df.trade_date = pd.to_datetime(df.trade_date).dt.date
    df = df.groupby(["trade_date", "maturity_bucket"])[["cvolu", "pvolu"]].sum()
    df = df.reset_index()
    df.index = df.trade_date
    df = df.sort_index()
    output_dir = "paper/res_paper/data_statistics/volume_count/"
    os.makedirs(output_dir, exist_ok=True)
    plotter = Plotter()
    for volume_type in ["pvolu", "cvolu"]:
        dataframes = []
        for maturity_bucket in MATURITY_BUCKETS:
            sub_df = df[df.maturity_bucket == maturity_bucket]
            sub_df = create_dataframe_for_smooth_plot(sub_df, columns=[volume_type])
            sub_df["maturity_bucket"] = maturity_bucket
            sub_df.date = pd.to_datetime(sub_df.date).dt.date
            dataframes.append(sub_df)

        plotter.plot_multiple_lines(
            dfs=dataframes,
            x="date",
            y=volume_type,
            xlabel="Year",
            ylabel="Count",
            save_name=os.path.join(
                output_dir, prefix_name + f"_{volume_type}_volume.png"
            ),
            labels=MATURITY_LABELS,
            title="",
            colors=None,
            linestyles=None,
        )
