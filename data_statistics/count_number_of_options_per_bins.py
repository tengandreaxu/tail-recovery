import argparse
import os
import time
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


def create_bins_counting_dataset(
    files_: list, output_dir: str, tickers_subset: str
) -> pd.DataFrame:
    """
    number of options in each bins
    """
    dataset = pd.DataFrame()

    tickers = []
    if tickers_subset:
        tickers = pd.read_csv(tickers_subset)
        tickers = tickers.ticker.tolist()

    for file_ in files_:
        start = time.monotonic()

        ticker = file_.split("/")[-1].replace(".p", "")
        if len(tickers) > 0 and ticker not in tickers:
            logging.info(f"Skipping {ticker} not in the small subset")
            continue

        df = pd.read_pickle(file_)
        df = df.reset_index(drop=True)
        df.pop("ticker")
        dataset = pd.concat([dataset, df])

        # save memory! giving up small cpu power
        dataset = dataset.groupby("trade_date").agg("mean")
        dataset["trade_date"] = dataset.index
        dataset = dataset.reset_index(drop=True)
        end = time.monotonic()
        process_time = round(end - start, 2)

        logging.info(
            "{:<35} {:<25}".format(f"Reading: {file_}", f"Time: {process_time}s")
        )
    dataset.to_pickle(os.path.join(output_dir, "counting_dataset.p"), protocol=4)
    return dataset


if __name__ == "__main__":
    """
    This scripts reads the bins count and plots the number of AVERAGE options per bins
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        type=str,
        dest="data_path",
        help="path to folder which contains the bins count",
    )

    parser.add_argument(
        "--is-big",
        action="store_true",
        dest="is_big",
        help="either is the small dataset or is the whole",
    )
    parser.set_defaults(is_big=False)

    parser.add_argument(
        "--already-computed",
        type=str,
        dest="already_computed",
        help="path to the already_compute file bins_count/counting_dataset.p",
    )

    parser.add_argument(
        "--tickers-subset",
        type=str,
        dest="tickers_subset",
        help="path to the tickers subset",
    )
    parser.set_defaults(tickers_subset=None)

    args = parser.parse_args()

    if not args.data_path and not args.already_computed:
        logging.error(
            "please specify at least one between data-path and already computed"
        )
        exit(1)

    prefix_name = "data_sample=small_"
    if args.is_big:
        prefix_name = "data_sample=big_"

    if not args.already_computed:
        files_ = os.listdir(args.data_path)
        files_ = [os.path.join(args.data_path, x) for x in files_ if ".p" in x]
        files_.sort()

        output_dir = os.path.join(args.data_path, "bins_count")
        os.makedirs(output_dir, exist_ok=True)
        df = create_bins_counting_dataset(files_, output_dir, args.tickers_subset)
    else:
        df = pd.read_pickle(args.already_computed)
    df.index = pd.to_datetime(df.trade_date).dt.date
    df.pop("trade_date")

    final_table = []
    for column in df.columns.tolist():
        if (
            column.startswith("pbidiv")
            or column.startswith("cbidiv")
            or column.startswith("open_interest")
            or column.startswith("volume")
        ):
            continue

        row = dict()
        option_type = "call"
        if "option_type=-1" in column:
            option_type = "put"
        row["right"] = option_type

        if "moneyness=0.0" in column:
            moneyness = "atm"
        elif "moneyness=1.0" in column:
            moneyness = "itm"
        elif "moneyness=2.0" in column:
            moneyness = "deep_itm"
        elif "moneyness=-1.0" in column:
            moneyness = "otm"
        else:
            moneyness = "deep_otm"

        row["moneyness"] = moneyness

        if "maturity_bucket=0.0" in column:
            maturity_bucket = 0
        elif "maturity_bucket=5.0" in column:
            maturity_bucket = 5.0

        elif "maturity_bucket=15.0" in column:
            maturity_bucket = 15.0
        elif "maturity_bucket=30.0" in column:
            maturity_bucket = 30.0
        elif "maturity_bucket=60.0" in column:
            maturity_bucket = 60.0
        elif "maturity_bucket=120.0" in column:
            maturity_bucket = 120.0
        else:
            maturity_bucket = 250.0

        row["maturity"] = maturity_bucket

        row["mean"] = round(df[column].mean(), 3)
        row["std"] = round(df[column].std(), 3)
        for quantile in [0.01, 0.25, 0.5, 0.75, 0.99]:
            row[f"q={quantile}"] = round(df[column].quantile(quantile), 2)

        final_table.append(row)
    caption = r"""The table shows the daily average number of options in a 
    bin bucket, the number of options standard deviation in 
    the time series, and the time series's quantiles for the following values: 
    0.01, 0.25, 0.50, 0.75, 0.99."""
    label = "table:bins_count"
    final_df = pd.DataFrame(final_table)
    final_df = final_df.sort_values("mean", ascending=False)
    output_dir = "paper/res_paper/data_statistics/bins_count/"
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_latex(
        f"{output_dir}{prefix_name}bins.tex",
        index=False,
        longtable=True,
        caption=caption,
        label=label,
    )
