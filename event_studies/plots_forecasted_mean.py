import argparse
import os

import pandas as pd

from util.dataframes_handling import load_all_pickles_in_folder
from plotting.Plotter import Plotter
from event_studies.event_study_earnings_announcements import get_only_earnings

YLABEL = "$E[y|\\xi,\sigma]$"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions-folder", dest="predictions_folder", type=str)

    parser.add_argument("--earnings-file", dest="earnings_file", type=str)

    parser.add_argument("--is-pos", dest="is_pos", action="store_true")
    parser.set_defaults(is_pos=False)

    args = parser.parse_args()
    df = load_all_pickles_in_folder(
        os.path.join(args.predictions_folder, "rolling_pred/")
    )
    plotter = Plotter()
    output_dir = os.path.join(args.predictions_folder, "scatter")
    os.makedirs(output_dir, exist_ok=True)
    correlation = df["y"].corr(df["pred"])
    correlation = round(correlation, 3)
    print(f"Correlation on jumps={correlation}")
    observation = df.shape[0]
    file_name = os.path.join(
        output_dir,
        f"mean_scatter_all_is_pos={args.is_pos}_corr={correlation}_obs={observation}.png",
    )
    plotter.scatter_plot(
        x=df["y"],
        y=df["pred"],
        ylabel=YLABEL,
        xlabel="$y_{i,t}$",
        grid=False,
        title="",
        color="black",
        marker="+",
        save_path=file_name,
    )

    earnings = pd.read_csv(args.earnings_file)
    df = get_only_earnings(df, earnings)
    df = df[df.is_earning_day == True]
    print(f"Jump on earnings: {df.shape[0]}")
    correlation = df["y"].corr(df["pred"])
    correlation = round(correlation, 3)
    print(f"Correlation on earnings={correlation}")
    observation = df.shape[0]
    file_name_earnings = os.path.join(
        output_dir,
        f"mean_scatter_earnings_only_is_pos={args.is_pos}_corr={correlation}_obs={observation}.png",
    )
    plotter.scatter_plot(
        x=df["y"],
        y=df["pred"],
        ylabel=YLABEL,
        xlabel="$y_{i,t}$",
        grid=False,
        title="",
        color="black",
        marker="+",
        save_path=file_name_earnings,
    )

    df = df.reset_index()
    df["year"] = pd.to_datetime(df.date).dt.year
    for year in df.year.unique().tolist():
        print(f"year: {year}")
        sub_df = df[df.year == year]
        print(f"Jump on earnings: {sub_df.shape[0]}")
        correlation = sub_df["y"].corr(sub_df["pred"])
        correlation = round(correlation, 3)
        print(f"Correlation on earnings={correlation}")
        observation = sub_df.shape[0]
        file_name_earnings = os.path.join(
            output_dir,
            f"mean_scatter_earnings_only_year={year}_is_pos={args.is_pos}_corr={correlation}_obs={observation}.png",
        )
        plotter.scatter_plot(
            x=sub_df["y"],
            y=sub_df["pred"],
            ylabel=YLABEL,
            xlabel="$y_{i,t}$",
            grid=False,
            title="",
            color="black",
            marker="+",
            save_path=file_name_earnings,
        )
