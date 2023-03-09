import os
import pandas as pd
from typing import Optional


def create_dataframe_for_smooth_plot_average(
    df: pd.DataFrame, columns: list
) -> pd.DataFrame:
    """
    the df has to be already ordered by date
    """
    output = []
    i = 0
    for index, row in df.iterrows():
        start = i - 252
        start = max(0, start)

        sub_df = df[start : (i + 1)]  # non inclusive

        row = dict()

        for column in columns:
            count = sub_df[column].mean()
            row[column] = count
        row["date"] = index
        output.append(row)
        i += 1
    to_plot = pd.DataFrame(output)
    to_plot = to_plot.sort_values("date")
    return to_plot


def create_dataframe_for_smooth_plot(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    the df has to be already ordered by date
    """
    output = []
    i = 0
    for index, row in df.iterrows():
        start = i - 252
        start = max(0, start)

        sub_df = df[start : (i + 1)]  # non inclusive

        row = dict()

        for column in columns:
            count = sub_df[column].sum()
            row[column] = count
        row["date"] = index
        output.append(row)
        i += 1
    to_plot = pd.DataFrame(output)
    to_plot = to_plot.sort_values("date")
    return to_plot


def load_all_pickles_in_folder(folder: str) -> pd.DataFrame:
    """
    Given a folder which contains pickles, this function returns
    a DataFrame of all pickles
    """
    files = os.listdir(folder)
    files.sort()

    df = pd.concat(
        [pd.read_pickle(os.path.join(folder, x)) for x in files if ".p" in x]
    )
    df.reset_index(drop=True)
    return df


def load_all_csvs_in_folder(
    folder: str, do_not_check_csv: Optional[bool] = False
) -> pd.DataFrame:
    """
    Given a folder which contains csvs, this function returns
    a DataFrame of all csvs
    """
    files = os.listdir(folder)
    files.sort()

    if do_not_check_csv:
        df = pd.concat([pd.read_csv(os.path.join(folder, x)) for x in files])
    else:
        df = pd.concat(
            [pd.read_csv(os.path.join(folder, x)) for x in files if ".csv" in x]
        )
    df.reset_index(drop=True)
    return df
