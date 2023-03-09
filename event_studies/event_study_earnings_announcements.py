import os
import argparse

import pandas as pd
import multiprocessing as mp

from datetime import datetime
from util.dataframes_handling import load_all_pickles_in_folder
from event_studies.event_study import (
    build_average_matrix,
    average_over_columns,
    EVENT_JUMPS_FOLDER,
    plot_time_series,
)


def get_only_earnings(
    full_predictions: pd.DataFrame, earnings_announcements: pd.DataFrame
) -> pd.DataFrame:
    """
    Given
     - a dataframe of predictions :df
     - a dataframe of earnings :earnings
    Returns
    - the dataframe of predictions during earnings only
    """
    predictions_tickers = full_predictions.ticker.tolist()
    event = "is_earning_day"
    earnings_announcements[event] = True
    earnings_announcements = earnings_announcements.rename(columns={"earnDate": "date"})
    earnings_announcements = earnings_announcements[~earnings_announcements.date.isna()]
    earnings_announcements = earnings_announcements[
        earnings_announcements.ticker.isin(predictions_tickers)
    ]
    earnings_announcements["date"] = pd.to_datetime(earnings_announcements.date).dt.date
    earnings_announcements = earnings_announcements[
        earnings_announcements.date > datetime.fromisoformat("2007-01-01").date()
    ]
    earnings_announcements.index = earnings_announcements[["ticker", "date"]]
    earnings_announcements.pop("ticker")
    earnings_announcements.pop("date")
    earnings_announcements.pop("anncTod")

    full_predictions.date = pd.to_datetime(full_predictions.date).dt.date
    full_predictions.index = full_predictions[["ticker", "date"]]

    grouped = full_predictions.join(earnings_announcements, how="left")
    grouped.loc[~(grouped.is_earning_day == True), event] = False
    return grouped


def time_series_during_earning_days(
    full_predictions: pd.DataFrame,
    earnings_announcements: pd.DataFrame,
    is_pos: bool,
    output_dir: str,
    pool: mp.Pool,
    range_min: int,
    range_max: int,
):
    """ """
    grouped = get_only_earnings(full_predictions, earnings_announcements)
    average_matrix = build_average_matrix(grouped, event=event, pool=None)
    time_series = average_over_columns(average_matrix)
    time_series_name = f"time_series_event={event}_is_positive={is_pos}.p"
    time_series.to_pickle(f"{output_dir}{time_series_name}")
    return time_series


if __name__ == "__main__":
    """
    same as event_study but for earnings_announcements
    days
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full-predictions",
        dest="full_predictions",
        type=str,
        help="path to the full predictions folder",
        required=False,
    )
    parser.add_argument(
        "--earnings-announcements-file",
        dest="earnings_announcements",
        help="path to the earnings file",
        type=str,
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

    parser.add_argument("--t", type=int, dest="t", help="prediction horizon")
    parser.set_defaults(t=1)

    parser.add_argument(
        "--range-min", type=int, dest="range_min", help="range min for the timeseries"
    )
    parser.set_defaults(range_min=-10)

    parser.add_argument(
        "--range-max", type=int, dest="range_max", help="range max for the timeseries"
    )
    parser.set_defaults(range_max=11)

    parser.add_argument(
        "--time-series-earnings",
        dest="time_series_earnings",
        type=str,
        help="path to the already computed time series",
    )
    args = parser.parse_args()

    rolling_std = f"rolling_historical_std_zero_mean_window={252}"
    output_dir = EVENT_JUMPS_FOLDER
    os.makedirs(output_dir, exist_ok=True)
    event = "is_earning_day"

    if args.time_series_earnings:
        time_series = pd.read_pickle(args.time_series_earnings)
    else:
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
                f"return_t={args.t}",
                rolling_std,
            ]
        ]
        predictions_tickers = full_predictions.ticker.tolist()

        earnings_announcements = pd.read_csv(args.earnings_announcements)
        if args.parallel:
            pool = mp.Pool(40)
            time_series = time_series_during_earning_days(
                full_predictions,
                earnings_announcements,
                args.is_pos,
                output_dir,
                pool,
                args.range_min,
                args.range_max,
            )
            pool.close()
            pool.join()
        else:
            time_series = time_series_during_earning_days(
                full_predictions,
                earnings_announcements,
                args.is_pos,
                output_dir,
                None,
                args.range_min,
                args.range_max,
            )
    plot_time_series(time_series, output_dir, event=event, is_pos=args.is_pos)
