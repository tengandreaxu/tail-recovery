import os
import argparse

import pandas as pd
import numpy as np

from datetime import datetime

from plotting.Plotter import Plotter


import logging

logging.basicConfig(level=logging.INFO)

PUT_OTM = ["put_moneyness=-1.0", "put_moneyness=-2.0"]
CALL_OTM = ["call_moneyness=-1.0", "call_moneyness=-2.0"]

PUT_OTM_ATM = ["put_moneyness=-1.0", "put_moneyness=-2.0", "put_moneyness=0.0"]
CALL_OTM_ATM = ["call_moneyness=-1.0", "call_moneyness=-2.0", "call_moneyness=0.0"]
PLOTS = [(PUT_OTM, "put_otm"), (CALL_OTM, "call_otm")]


def plot_the_average_of_otm_call_and_average_otm_puts(
    df: pd.DataFrame, output_dir: str
):

    output = output_dir + "/signs_of_coefficients/"
    os.makedirs(output, exist_ok=True)

    plotter = Plotter()
    try:
        to_plot = df[CALL_OTM].copy()
        to_plot["mean"] = to_plot.mean(axis=1)
        plotter.plot_single_curve(
            to_plot.index,
            to_plot["mean"],
            ylabel="Average sign of coefficients",
            xlabel="Year",
            grid=True,
            title="",
            save_path=f"{output}call_otm_average.png",
            color="black",
            linestyle="solid",
            label="Average OTM Calls",
        )
    except:
        logging.info(f"{CALL_OTM} not in index")
    try:
        to_plot = df[PUT_OTM].copy()
        to_plot["mean"] = to_plot.mean(axis=1)
        plotter.plot_single_curve(
            to_plot.index,
            to_plot["mean"],
            ylabel="Average sign of coefficients",
            xlabel="Year",
            grid=True,
            title="",
            save_path=f"{output}put_otm_average.png",
            color="black",
            linestyle="solid",
            label="Average OTM Puts",
        )
    except:
        logging.info(f"{PUT_OTM} not in index")


def plot_the_sign_of_coefficients(
    df: pd.DataFrame, columns: list, name: str, output_dir: str
):
    output = output_dir + "signs_of_coefficients/"
    os.makedirs(output, exist_ok=True)

    plotter = Plotter()
    try:
        to_plot = df[columns]
        plotter.plot_multiple_lines_from_columns(
            df=to_plot,
            title="",
            ylabel="Average sign of coefficients",
            xlabel="Year",
            save_name=f"{name}.png",
            output_dir=output,
        )
    except:
        logging.info(f"{columns} not in index")


def plot_the_average_of_betas(
    df: pd.DataFrame, columns: list, name: str, output_dir: str
):

    output = output_dir + "average_betas/"
    os.makedirs(output, exist_ok=True)

    plotter = Plotter()
    try:
        to_plot = df[columns]
        plotter.plot_multiple_lines_from_columns(
            df=to_plot,
            title="",
            ylabel="Average coefficients",
            xlabel="Year",
            save_name=f"{name}.png",
            output_dir=output,
        )
    except:
        logging.info(f"{columns} not in index")


def rolling_year(df: pd.DataFrame, agg: str, column: str):

    to_plot = pd.DataFrame()
    months = df["month"].unique().tolist()
    for i in range(12, len(months)):
        m_start = months[i - 12]
        m_end = months[i]
        sub = df[(df.month >= m_start) & (df.month <= m_end)]

        sub = sub.groupby(["predictor", "month"]).agg(agg)[column].reset_index()
        sub = sub.groupby("predictor").agg("mean")

        sub = pd.DataFrame(sub).T
        logging.info(f"month={m_end}, other_historical={sub.iloc[0].other_historical}")
        sub.index = [m_end]
        to_plot = pd.concat([to_plot, sub])
    return to_plot


def clean_predictors_name(df: pd.DataFrame) -> pd.DataFrame:
    """ """
    predictors = df.predictor_name.tolist()
    predictors = [x.split("_") for x in predictors]

    cleaned = []

    ## we are interested on (option_type, moneyness) only
    for predictor in predictors:
        # e.g. rolling historical std zero mean window 252
        call_or_put = "other"
        type_ = predictor[-1]
        if type_ == "type=1":
            call_or_put = "call"
        if type_ == "type=-1":
            call_or_put = "put"

        if type_ != "other":
            moneyness = predictor[1]
        else:
            moneyness = ""
        cleaned.append(call_or_put + "_" + moneyness)
    df["predictor"] = cleaned
    return df


def load_ols_coefficients(directory: str) -> pd.DataFrame:
    """
    Given a result folder, returns
    the ols parameters coefficients as df
    and keep only relevant predictors
    """

    dir_ = directory + "/rolling_ols_parameters/"
    files_ = os.listdir(dir_)

    df = pd.DataFrame()
    for file_ in files_:
        sub_df = pd.read_pickle(os.path.join(dir_, file_))

        sub_df = sub_df.rename(columns={"P>|t|": "p_value"})
        sub_df["month"] = datetime.strptime(file_.replace(".p", ""), "%Y%m").date()
        df = pd.concat([df, sub_df])
    df = df[df.p_value < 0.05]
    df = df.sort_values("month")
    return df


if __name__ == "__main__":
    """

     - redo the plots where you have  the time series of number of singificant
     parameters per option category type with new ys in seperate plots:

        i) y = average of beta coefficient
        ii) y = average of sign of the coefficient (so basically transform betas into -1 and +1 and compute average)

    same rolling window plots Do this plot --> do everything OTM only:
        a) on the one we have in the paper
        b) on the one where you do call only regression and put only regression
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--coeff-folder",
        dest="coeff_folder",
        type=str,
        help="path to the coefficients folder",
    )

    args = parser.parse_args()

    df = load_ols_coefficients(args.coeff_folder)
    df = clean_predictors_name(df)
    df["coef_sign"] = np.sign(df.coef)
    average_of_sign = rolling_year(df.copy(), agg="mean", column="coef_sign")
    average_of_beta = rolling_year(df.copy(), agg="mean", column="coef")
    for columns, name in PLOTS:
        plot_the_sign_of_coefficients(average_of_sign, columns, name, args.coeff_folder)
        plot_the_average_of_betas(average_of_beta, columns, name, args.coeff_folder)
        plot_the_average_of_otm_call_and_average_otm_puts(
            average_of_sign, args.coeff_folder
        )
