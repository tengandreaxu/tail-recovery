import argparse
import holidays
import os
import time

import numpy as np
import pandas as pd
import multiprocessing as mp

from datetime import datetime, date
from pandas.tseries.holiday import USFederalHolidayCalendar

from util.dataframes_handling import load_all_pickles_in_folder
from plotting.Plotter import Plotter
from plotting.plot_constants import MODEL_PRETTY_NAMES, MODEL_PRETTY_LABELS


import logging

logging.basicConfig(level=logging.INFO)

US_HOLIDAYS = holidays.US()
EVENT_JUMPS_FOLDER = "paper/res_paper/event_studies/jumps/"


def jumps_indexes(df: pd.DataFrame, std: float, t: int):

    qv = df["rolling_historical_std_zero_mean_window=252"] * np.sqrt(t)
    if std > 0:
        indexing = df[f"return_t={t}"] >= qv * std
    else:
        indexing = df[f"return_t={t}"] <= qv * std
    return indexing


def list_of_business_days_between(start_date: date, end_date: date):
    """
    This function returns the list of business days between
    two dates
    """
    bday_us = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())

    bdays_range = pd.date_range(start_date, end_date, freq=bday_us)
    return bdays_range


def plot_time_series(
    time_series: pd.DataFrame, output_dir: str, event: str, is_pos: bool
):
    """ """
    plotter = Plotter()
    columns = ["sigma_model", "xi_model", "s_pred", "pred"]
    days = list(range(-10, 11))
    for column in columns:
        to_plot = []
        for day in days:
            to_plot.append(
                {"day": day, "column": time_series[f"{column}_t={day}"].values[0]}
            )
        to_plot = pd.DataFrame(to_plot)
        plotter.plot_single_curve(
            x=to_plot.day,
            y=to_plot.column,
            title="",
            ylabel=MODEL_PRETTY_NAMES[column],
            xlabel="T",
            grid=True,
            save_path=f"{output_dir}is_positive={is_pos}_event={event}_{column}.png",
            color="black",
            linestyle="solid",
            label=MODEL_PRETTY_LABELS[column],
        )


def average_over_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    General function, fixing a column,
    average over rows and returns the dataframe
    """
    columns = df.columns.tolist()
    output = dict()
    for column in columns:
        output[column] = np.nanmean(df[column])
    output = pd.DataFrame(output, index=[0])
    return output


def safe_add(mini_matrix: dict, row: pd.Series, day: int):

    columns = ["sigma_model", "xi_model", "s_pred", "pred"]

    for column in columns:
        column_matrix = f"{column}_t={day}"

        if column_matrix in mini_matrix:
            mini_matrix[column_matrix].append(row[column])
        else:
            mini_matrix[column_matrix] = [row[column]]
    return mini_matrix


def add_missing_business_days(
    df: pd.DataFrame, ticker: str, event: str
) -> pd.DataFrame:
    """
    Because we predict only on those dates where
    we have enough data.
    """
    start_date = df.date.min()
    end_date = df.date.max()
    business_days = list_of_business_days_between(start_date, end_date)

    to_append = pd.DataFrame(business_days, columns=["date"])
    to_append["ticker"] = ticker
    to_append[event] = False

    for column in ["xi_model", "sigma_model", "pred", "s_pred"]:
        to_append[column] = np.nan

    to_append.date = pd.to_datetime(to_append.date).dt.date
    output = pd.concat([df, to_append])
    output = output.sort_values("date")
    output = output.reset_index(drop=True)
    return output


def compute_average_matrix_per_ticker(
    ticker: str,
    event: str,
    sub_df: pd.DataFrame,
    range_min: int = -10,
    range_max: int = 11,
) -> pd.DataFrame:

    mini_matrix = dict()
    logging.info(f"ticker={ticker}, event={event}")
    start = time.monotonic()
    sub_df = add_missing_business_days(sub_df, ticker, event)
    available_indexes = sub_df.index.tolist()
    event_days = sub_df[sub_df[event] == True].index.tolist()

    days = list(range(range_min, range_max))
    ## really can't avoid O(n^2)??
    for index in event_days:
        for day in days:
            if (index + day) in available_indexes:
                mini_matrix = safe_add(mini_matrix, sub_df.loc[(index + day)], day)
    ticker_average = pd.DataFrame(
        {key: pd.Series(value) for key, value in mini_matrix.items()}
    )
    # ticker_average = ticker_average.fillna(value=0)

    ticker_average = average_over_columns(ticker_average)
    end = time.monotonic()
    logging.info(f"Average time per ticker: {round(end-start, 2)}s")
    return ticker_average


def build_average_matrix(full_predictions: pd.DataFrame, event: str, pool: mp.Pool):
    """
    Builds the average matrix for each ticker,
    then aggregates
    """
    tickers = full_predictions.ticker.unique().tolist()
    output = []
    if not pool:
        for ticker in tickers:

            sub_df = full_predictions[full_predictions.ticker == ticker]
            ticker_average = compute_average_matrix_per_ticker(ticker, event, sub_df)
            output.append(ticker_average)
    else:
        output += pool.starmap(
            compute_average_matrix_per_ticker,
            [
                (
                    ticker,
                    event,
                    full_predictions[full_predictions.ticker == ticker].copy(),
                )
                for ticker in tickers
            ],
        )

    final = pd.concat(output)
    return final


def time_series_during_jump_days(
    full_predictions: pd.DataFrame, t: int, is_pos: bool, pool: mp.Pool
):
    """ """
    std = -2.0
    if is_pos:
        std = 2.0
    event = "is_jump_day"
    jump_days = jumps_indexes(full_predictions, std, t)
    full_predictions.loc[jump_days, event] = True
    full_predictions.loc[~jump_days, event] = False

    full_predictions.pop(rolling_std)
    full_predictions.pop("y")
    full_predictions.pop("return_t=1")
    full_predictions.date = pd.to_datetime(full_predictions.date).dt.date

    average_matrix = build_average_matrix(full_predictions, event=event, pool=pool)
    # average_matrix = average_matrix.dropna()

    time_series = average_over_columns(average_matrix)
    output_dir = EVENT_JUMPS_FOLDER
    os.makedirs(output_dir, exist_ok=True)
    time_series_name = f"time_series_event={event}_is_positive={is_pos}.p"
    time_series.to_pickle(f"{output_dir}{time_series_name}")
    plot_time_series(time_series, output_dir, event=event, is_pos=is_pos)


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full-predictions",
        dest="full_predictions",
        type=str,
        help="path to the full predictions folder",
        required=True,
    )
    parser.add_argument(
        "--is-pos",
        dest="is_pos",
        action="store_true",
        help="either is the positive tail or not",
    )
    parser.set_defaults(is_pos=False)
    parser.add_argument("--parallel", action="store_true", dest="parallel")
    parser.set_defaults(parallel=False)

    parser.add_argument(
        "--time-series-jump",
        dest="time_series_jump",
        type=str,
        help="path to the already computed time series",
    )
    parser.add_argument("--t", dest="t", type=int, help="prediction horizon")
    args = parser.parse_args()

    rolling_std = f"rolling_historical_std_zero_mean_window={252}"

    full_predictions = load_all_pickles_in_folder(args.full_predictions)
    full_predictions = full_predictions[
        [
            "xi_model",
            "sigma_model",
            "ticker",
            "date",
            "s_pred",
            "pred",
            "y",
            "return_t=1",
            rolling_std,
        ]
    ]

    if args.parallel:
        pool = mp.Pool(30)
    else:
        pool = None

    if args.time_series_jump:
        time_series = pd.read_pickle(args.time_series_jump)
        plot_time_series(
            time_series,
            output_dir=EVENT_JUMPS_FOLDER,
            event="is_jump_day",
            is_pos=args.is_pos,
        )
    else:
        time_series_during_jump_days(
            full_predictions.copy(), args.t, args.is_pos, pool=pool
        )

    # time_series_during_earning_days(full_predictions, earnings_announcements)
    if args.parallel:
        pool.close()
        pool.join()
