import argparse
import os

import pandas as pd


def copy_files(output_dir: str, tickers: list, input_folder: str):

    os.makedirs(output_dir, exist_ok=True)
    for ticker in tickers:
        file_name = ticker + ".csv"
        print(f"Reading {file_name}")
        try:
            df = pd.read_csv(os.path.join(input_folder, file_name))
        except:
            print(f"missing {file_name}")
            continue
        df.to_csv(os.path.join(output_dir, file_name), index=False)


if __name__ == "__main__":
    """In this script,
    we perform subsetting of the parent population by creating a smaller,
    representative subset."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--big-grids", dest="big_grids", type=str, help="path to the big dataset grid"
    )

    parser.add_argument(
        "--tickers-subset",
        dest="tickers_subset",
        type=str,
        help="a csv with a small subset of csv, we use the S&P500 subset",
    )

    parser.add_argument(
        "--raw-option",
        dest="raw_option",
        type=str,
        help="path to the big dataset raw option data",
    )

    parser.add_argument(
        "--grids-destination",
        dest="grids_destination",
        type=str,
        help="where to copy the small dataset grids",
    )

    parser.add_argument(
        "--raw-option-destination",
        dest="raw_option_destination",
        type=str,
        help="where to copy the small dataset raw option data",
    )
    args = parser.parse_args()

    tickers = pd.read_csv(args.tickers_subset)
    tickers = tickers.ticker.tolist()

    copy_files(args.grids_destination, tickers, args.big_grids)
    copy_files(
        args.raw_option_destination,
        tickers,
        args.raw_option,
    )
