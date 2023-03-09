import argparse
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from plotting.plot_functions import (
    format_datetime_into_years,
)
from util.dataframes_handling import create_dataframe_for_smooth_plot
from plotting.plot_function import pylab
import logging

logging.basicConfig(level=logging.INFO)


def create_counting_dataset(files_: list, output_dir: str) -> pd.DataFrame:
    """
    Reads all files and counts number of options per day
    """
    dataset = pd.DataFrame()

    for file_ in files_:
        start = time.monotonic()
        df = pd.read_csv(file_)
        df = df.groupby("trade_date")["ticker"].agg("count")
        df = pd.DataFrame(df).rename(columns={"ticker": "count"})
        dataset = pd.concat([dataset, df])
        end = time.monotonic()
        process_time = round(end - start, 2)

        logging.info(
            "{:<35} {:<25}".format(f"Reading: {file_}", f"Time: {process_time}s")
        )
    dataset.to_pickle(os.path.join(output_dir, "counting_dataset.p"), protocol=4)
    return dataset


if __name__ == "__main__":
    """
    This scripts reads the raw datasets
    and computes daily number of options
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        type=str,
        dest="data_path",
        help="path to folder which contains raw orats data",
    )

    parser.add_argument(
        "--counting-dataset-file",
        type=str,
        dest="counting_file",
        help="path to the already computed counting dataset",
    )

    parser.add_argument(
        "--is-big",
        action="store_true",
        dest="is_big",
        help="either is the small dataset or is the whole",
    )
    parser.set_defaults(is_big=False)

    args = parser.parse_args()

    if not args.data_path and not args.counting_file:
        logging.error("please specify either data-path or counting-file")
        exit(1)

    prefix_name = "data_sample=small_"
    if args.is_big:
        prefix_name = "data_sample=big_"

    if args.data_path:
        files_ = os.listdir(args.data_path)
        files_ = [os.path.join(args.data_path, x) for x in files_ if ".csv" in x]
        files_.sort()

        output_dir = os.path.join(args.data_path, "number_of_options")
        os.makedirs(output_dir, exist_ok=True)
        df = create_counting_dataset(files_, output_dir)
    else:
        df = pd.read_pickle(args.counting_file)

    df["trade_date"] = df.index
    df.trade_date = pd.to_datetime(df.trade_date).dt.date
    df = df.reset_index(drop=True)
    df = df.groupby("trade_date")["count"].sum()
    df = df.sort_index()
    df = pd.DataFrame(df)
    df = create_dataframe_for_smooth_plot(df, columns=["count"])

    fig, ax = plt.subplots()

    ax.plot(
        df["date"],
        df["count"],
        label="Number of Options",
        color="black",
        linestyle="solid",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Options")

    ax.grid()
    ax.legend()
    format_datetime_into_years(ax)
    plt.tight_layout()
    output_folder = "paper/res_paper/data_statistics/options_count"
    os.makedirs(output_folder, exist_ok=True)
    file_ = os.path.join(output_folder, f"{prefix_name}number_of_options_years.png")
    fig.savefig(file_)
