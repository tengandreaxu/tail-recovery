import argparse
import os
import pandas as pd
import time
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import date
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=35)

import logging

logging.basicConfig(level=logging.INFO)


def business_days_between(start_date: date, end_date: date):
    """number of daily returns between start_date and end_date"""
    bday_us = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())

    bdays_range = pd.date_range(start_date, end_date, freq=bday_us)
    number_bdays = len(bdays_range[:-1])
    return number_bdays


def compute_business_days_to_maturity(folder: str, df: pd.DataFrame, old: pd.DataFrame):
    start_time = time.time()
    df = df.groupby(["trade_date", "expirdate"]).first()
    df = df.reset_index()
    logging.info(f"total number of rows: {df.shape[0]}")
    df.trade_date = pd.to_datetime(df.trade_date).dt.date
    df.expirdate = pd.to_datetime(df.expirdate).dt.date
    df["business_days_to_maturity"] = df.parallel_apply(
        lambda x: business_days_between(x.trade_date, x.expirdate), axis=1
    )
    end_time = time.time()
    logging.info(
        f"computing business days for {df.shape[0]} rows took {end_time - start_time}s"
    )

    df = df[["trade_date", "expirdate", "business_days_to_maturity"]]
    if not old.empty:
        df = pd.concat([df, old])
    df = df.reset_index(drop=True)
    destination_folder = "/".join(folder.split("/")[:-1])
    os.makedirs(f"{destination_folder}/underlyings_info/", exist_ok=True)
    df.to_csv(
        f"{destination_folder}/underlyings_info/business_days_to_maturity.csv",
        index=False,
    )


if __name__ == "__main__":
    """
    This script adds to ORATS raw data
    the business days to expiry,

    this feature is used later to get the ATM average iv
    for maturity_days in [1,5,20,252] to fit the regressor
    competing with Generalized Recovery paper
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--option-data-folder",
        dest="option_data_folder",
        type=str,
        help="path to orats raw option data",
        required=True,
    )

    parser.add_argument(
        "--add-missing-to",
        dest="add_missing_to",
        type=str,
        help="old business days to maturity to be updated",
    )
    parser.set_defaults(add_missing_to=None)
    args = parser.parse_args()

    options_data = [x for x in os.listdir(args.option_data_folder) if ".csv" in x]
    options_data.sort()

    aggregate = pd.DataFrame()
    count = len(options_data)
    for option_data in options_data:
        logging.info(f"Reading ticker={option_data}")
        df = pd.read_csv(f"{args.option_data_folder}/{option_data}")
        df = df.groupby(["trade_date", "expirdate"]).first()
        df = df.reset_index()
        df = df[["trade_date", "expirdate"]]
        aggregate = pd.concat([aggregate, df])
        count -= 1
        logging.info(f"missing = {count}")

    # compute only subset
    old = pd.DataFrame()
    if args.add_missing_to:
        old = pd.read_csv(args.add_missing_to)
        old.index = old[["trade_date", "expirdate"]]
        aggregate.index = aggregate[["trade_date", "expirdate"]]
        new = aggregate.join(old, rsuffix="_old", how="left")
        new = new[new.business_days_to_maturity.isna()]
        new = new.reset_index(drop=True)
        new.pop("trade_date_old")
        new.pop("expirdate_old")

        ##XXX kill me
        new = new[new.trade_date <= "2020-08-31"]
        aggregate = new
    compute_business_days_to_maturity(args.option_data_folder, aggregate, old)
