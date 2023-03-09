import argparse
import os
import pandas as pd
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
from parameters.parameters import Params
from training.trainer import Trainer, PRETTY_LABELS
from util.dataframes_handling import load_all_pickles_in_folder
from plotting.Plotter import Plotter


def analyze_performance_for_different_results(
    trainer: Trainer,
    res1_df: pd.DataFrame,
    res2_df: pd.DataFrame,
    res3_df: pd.DataFrame,
    lag: bool,
    is_pos: bool,
    otm_only: bool,
    t: int,
    output_dir: str,
):
    res1, res2, res3 = trainer.load_benchmark(
        res1_df,
        res2_df,
        res3_df,
        with_lag=lag,
        is_pos=is_pos,
    )

    os.makedirs(output_dir, exist_ok=True)
    # Call Only
    call_r2 = trainer.r2_f(res1)
    # Put Only
    put_r2 = trainer.r2_f(res2)
    # Both
    both_r2 = trainer.r2_f(res3)

    rolling1 = trainer.compute_rolling_r2(res1)
    rolling2 = trainer.compute_rolling_r2(res2)
    rolling3 = trainer.compute_rolling_r2(res3)

    ylim = [-0.055, 0.33]
    if args.t == 5:
        ylim = [-0.2, 0.3]

    plotter = Plotter()
    for column in rolling1.columns:
        r2 = column.replace("^", "").replace(", ", "_")
        file_name = f"t={t}_is_pos={is_pos}_otm_only={otm_only}_r2={r2}.png"

        csv_name = f"t={t}_is_pos={is_pos}_otm_only={otm_only}"
        call_r2.to_csv(os.path.join(output_dir, f"{csv_name}_call_only.csv"))
        put_r2.to_csv(os.path.join(output_dir, f"{csv_name}_put_only.csv"))
        both_r2.to_csv(os.path.join(output_dir, f"{csv_name}_both_only.csv"))

        ## NNET
        plotter.plot_multiple_lines(
            dfs=[rolling1, rolling2, rolling3],
            x="index",
            y=column,
            title="",
            xlabel="Year",
            ylabel=PRETTY_LABELS[column],
            save_name=os.path.join(output_dir, file_name),
            labels=["Call Only", "Put Only", "Both"],
            colors=["black", "brown", "navy"],
            linestyles=[":", "--", "-"],
            ylim=ylim,
        )


if __name__ == "__main__":
    """
    This script plots
    the different r2 performance achieved from different models
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--res-1",
        "-r1",
        dest="res_1",
        type=str,
        help="folder to the first res",
        required=True,
    )

    parser.add_argument(
        "--res-2",
        "-r2",
        dest="res_2",
        type=str,
        required=True,
        help="folder to the second res`",
    )
    parser.add_argument(
        "--res-3",
        "-r3",
        dest="res_3",
        type=str,
        required=False,
        help="folder to the third res`",
    )

    parser.add_argument(
        "--output-dir", "-od", dest="output_dir", type=str, help="output dir"
    )
    parser.set_defaults(output_dir="")

    parser.add_argument("--t", dest="t", type=int)
    parser.set_defaults(t=1)

    parser.add_argument(
        "--is-pos",
        "-ps",
        action="store_true",
        help="either positive or negative std",
        dest="is_pos",
    )
    parser.set_defaults(is_positive=False)

    parser.add_argument(
        "--otm-only", dest="otm_only", action="store_true", help="otm-only"
    )
    parser.set_defaults(otm_only=False)
    parser.add_argument("--lag", action="store_true", dest="lag")
    parser.set_defaults(lag=False)

    parser.add_argument(
        "--predictions-dir",
        dest="predictions_dir",
        type=str,
        help="path to the predictions folder",
    )
    parser.set_defaults(predictions_dir="res")

    args = parser.parse_args()

    # load r2 results from the BIG dataset
    try:
        res1_df = load_all_pickles_in_folder(args.res_1)
        res2_df = load_all_pickles_in_folder(args.res_2)
        if args.res_3:
            res3_df = load_all_pickles_in_folder(args.res_3)
    except:
        res1_df = load_all_pickles_in_folder(
            os.path.join(args.predictions_dir, args.res_1, "rolling_pred/")
        )
        res2_df = load_all_pickles_in_folder(
            os.path.join(args.predictions_dir, args.res_2, "rolling_pred/")
        )
        if args.res_3:
            res3_df = load_all_pickles_in_folder(
                os.path.join(args.predictions_dir, args.res_3, "rolling_pred/")
            )

    par = Params()
    par.data.t = args.t
    np.random.seed(par.seed)
    tf.random.set_seed(par.seed)
    trainer = Trainer(par)

    output_dir = "paper/res_paper/otm_study/"
    analyze_performance_for_different_results(
        trainer,
        res1_df,
        res2_df,
        res3_df,
        args.lag,
        args.is_pos,
        args.otm_only,
        args.t,
        output_dir,
    )
