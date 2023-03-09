import os
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from plotting.plot_function import pylab
from plotting.plot_functions import (
    get_xticks,
    format_datetime_into_years,
)

import logging

logging.basicConfig(level=logging.INFO)


def flat_list(list_of_list: list) -> list:

    flat_list = [item for sublist in list_of_list for item in sublist]
    return flat_list


def compute_rolling_number_of_tickers(output_dir: str) -> pd.DataFrame:
    """
    computes for each date the rolling windwos number of tickers
    """

    df = pd.read_pickle(args.train_file)

    df["date"] = df.index
    grouped = df.groupby("date")["ticker"].apply(list)

    grouped = pd.DataFrame(grouped)
    grouped = grouped.sort_index()

    output = []
    i = 0
    for index, row in grouped.iterrows():
        start = i - 252
        start = max(0, start)

        sub_df = grouped[start : (i + 1)]  # non inclusive
        tickers = sub_df.ticker.tolist()

        tickers = flat_list(tickers)
        tickers = set(tickers)
        output.append({"date": index, "num_tickers": len(tickers)})
        i += 1
    to_plot = pd.DataFrame(output)
    to_plot = to_plot.sort_values("date")
    os.makedirs(output_dir, exist_ok=True)

    file_pickle = f"{output_dir}/{prefix_name}_tickers_rolling_window_count.p"
    to_plot.to_pickle(file_pickle)
    return to_plot


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-ticker-file",
        "-tf",
        type=str,
        help="path to the train_ticker.p file",
        dest="train_file",
    )

    parser.add_argument(
        "--is-big",
        action="store_true",
        help="is either the big dataset or not",
        dest="is_big",
    )
    parser.set_defaults(is_big=False)

    parser.add_argument(
        "--computed-file",
        dest="computed_file",
        type=str,
        help="the already computed count",
    )

    args = parser.parse_args()

    output_dir = "paper/res_paper/data_statistics/tickers_count"
    os.makedirs(output_dir, exist_ok=True)
    prefix_name = "data_sample=small"
    if args.is_big:
        prefix_name = "data_sample=big"

    if not args.computed_file and args.train_file:
        to_plot = compute_rolling_number_of_tickers(output_dir)
    elif args.computed_file:
        to_plot = pd.read_pickle(args.computed_file)
    else:
        logging.error("Specify at least one between train_file or computed_file")
    file_name = os.path.join(
        output_dir, f"{prefix_name}_tickers_rolling_window_count.png"
    )
    fig, ax = plt.subplots()
    to_plot.date = pd.to_datetime(to_plot.date).dt.date
    ax.plot(
        to_plot["date"],
        to_plot["num_tickers"],
        color="black",
        label="Number of Tickers",
    )
    ax.set_ylabel("Number of Tickers")
    ax.set_xlabel("Year")
    ax.legend()
    xticks = get_xticks(to_plot.date.unique().tolist())
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=10)

    ax.grid()

    format_datetime_into_years(ax)
    # plt.show()
    # because it will go together
    # with options count and volume count
    # which have scientific notation
    ax.ticklabel_format(style="scientific", axis="y")

    plt.tight_layout()
    fig.savefig(file_name)
