import os
import argparse
from typing import Optional
import pandas as pd
import numpy as np

from plotting.Plotter import Plotter
from analysis.analysis_sign_of_coefficients import (
    load_ols_coefficients,
    clean_predictors_name,
    PUT_OTM,
    CALL_OTM,
    PUT_OTM_ATM,
    CALL_OTM_ATM,
)
import logging

logging.basicConfig(level=logging.INFO)


def smooth_plot(df: pd.DataFrame):

    to_plot = []

    rows = df.index.tolist()

    for i in range(0, len(rows)):
        if i < 12:
            m_start = rows[0]
        else:
            m_start = rows[i - 12]
        m_end = rows[i]
        sub = df[(df.index >= m_start) & (df.index <= m_end)]

        rolling_mean = np.nanmean(sub)
        to_plot.append({"month": m_end, "coef": rolling_mean})
    to_plot = pd.DataFrame(to_plot)
    to_plot.index = pd.to_datetime(to_plot.month).dt.date
    to_plot.pop("month")
    return to_plot


def sum_otm_with_deep_otm(
    df: pd.DataFrame, otm: str, deep_otm: str, atm: Optional[str] = ""
) -> pd.DataFrame:

    otm = df[df.predictor == otm].copy()
    deep_otm = df[df.predictor == deep_otm].copy()
    deep_otm = deep_otm.groupby("month")["coef"].agg("sum")
    otm = otm.groupby("month")["coef"].agg("sum")

    if atm != "":
        atm = df[df.predictor == atm].copy()
        atm = atm.groupby("month")["coef"].agg("sum")

        return deep_otm + otm + atm
    else:
        return deep_otm + otm


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--call-only",
        dest="call_only",
        type=str,
        help="path to the coefficients folder call only",
    )

    parser.add_argument(
        "--put-only",
        dest="put_only",
        type=str,
        help="path to the coefficients folder put only",
    )
    parser.add_argument(
        "--is-pos", dest="is_pos", action="store_true", help="either is pos or not"
    )
    parser.set_defaults(is_pos=False)
    args = parser.parse_args()

    plotter = Plotter()

    df_call = load_ols_coefficients(args.call_only)
    df_put = load_ols_coefficients(args.put_only)

    df_call = clean_predictors_name(df_call)
    df_put = clean_predictors_name(df_put)

    df = pd.concat([df_call, df_put])
    put = sum_otm_with_deep_otm(df, PUT_OTM[0], PUT_OTM[1])
    call = sum_otm_with_deep_otm(df, CALL_OTM[0], CALL_OTM[1])
    put_smooth = smooth_plot(put)
    call_smooth = smooth_plot(call)
    output_dir = "paper/res_paper/otm_study/average_betas/"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"otm_sum_betas_smooth_is_pos={args.is_pos}"
    plotter.plot_multiple_lines(
        dfs=[put_smooth, call_smooth],
        x="index",
        y="coef",
        title="",
        xlabel="Year",
        ylabel=r"$\beta$",
        save_name=os.path.join(output_dir, file_name),
        labels=["Calls", "Puts"],
        colors=["black", "brown"],
        linestyles=None,
    )

    # further analysis
    put = sum_otm_with_deep_otm(df, PUT_OTM[0], PUT_OTM[1], PUT_OTM_ATM[2])
    call = sum_otm_with_deep_otm(df, CALL_OTM[0], CALL_OTM[1], CALL_OTM_ATM[2])
    put_smooth = smooth_plot(put)
    call_smooth = smooth_plot(call)
    file_name = f"otm_atm_sum_betas_smooth_is_pos={args.is_pos}"
    plotter.plot_multiple_lines(
        dfs=[put_smooth, call_smooth],
        x="index",
        y="coef",
        title="",
        xlabel="Year",
        ylabel=r"$\beta$",
        save_name=os.path.join(output_dir, file_name),
        labels=["Calls", "Puts"],
        colors=["black", "brown"],
        linestyles=None,
    )
