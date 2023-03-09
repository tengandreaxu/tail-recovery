import os
import argparse
import pandas as pd
from datetime import datetime
from analysis.analysis_sign_of_coefficients import (
    clean_predictors_name,
    PUT_OTM,
    CALL_OTM,
)

from analysis.analysis_otm_average_betas import smooth_plot
from plotting.Plotter import Plotter


def load_lasso_betas_df(path: str) -> pd.DataFrame:
    files_ = os.listdir(path)
    df = pd.DataFrame()
    for file_ in files_:
        sub_df = pd.read_csv(os.path.join(path, file_))

        sub_df["month"] = datetime.strptime(file_, "%Y%m").date()
        df = pd.concat([df, sub_df])

    df = df.sort_values("month")
    df = df.rename(columns={"columns": "predictor_name"})
    df = clean_predictors_name(df)

    df = df.groupby(["month", "predictor"]).sum().reset_index()
    df.index = df.month
    df.pop("month")
    return df


def get_smooth_time_series(df: pd.DataFrame, is_call: bool) -> pd.DataFrame:
    if is_call:
        otm = df[df.predictor == CALL_OTM[0]].copy()
        deep_otm = df[df.predictor == CALL_OTM[1]].copy()

    else:
        otm = df[df.predictor == PUT_OTM[0]].copy()
        deep_otm = df[df.predictor == PUT_OTM[1]].copy()

    otm.pop("predictor")
    deep_otm.pop("predictor")

    otm_smooth = smooth_plot(otm)
    deep_otm_smooth = smooth_plot(deep_otm)

    return otm_smooth + deep_otm_smooth


if __name__ == "__main__":
    plotter = Plotter()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lasso-coeff-call",
        dest="lasso_coeff_call",
        type=str,
        required=True,
        help="path to the lasso betas trained with otm call only",
    )

    parser.add_argument(
        "--lasso-coeff-put",
        dest="lasso_coeff_put",
        type=str,
        required=True,
        help="path to the lasso betas trained with otm put only",
    )

    parser.add_argument("--is-pos", dest="is_pos", action="store_true")
    parser.set_defaults(is_pos=False)

    args = parser.parse_args()
    df_call = load_lasso_betas_df(args.lasso_coeff_call)
    df_call = get_smooth_time_series(df_call, is_call=True)

    df_put = load_lasso_betas_df(args.lasso_coeff_put)
    df_put = get_smooth_time_series(df_put, is_call=False)

    output_dir = "paper/res_paper/otm_study/lasso_average_betas/"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"otm_avg_betas_smooth_is_pos={args.is_pos}"
    plotter.plot_multiple_lines(
        dfs=[df_call, df_put],
        x="index",
        y="coef",
        title="",
        xlabel="Year",
        ylabel=r"$\beta$",
        save_name=os.path.join(output_dir, file_name),
        labels=["Call", "Put"],
        colors=["black", "brown"],
        linestyles=None,
    )
