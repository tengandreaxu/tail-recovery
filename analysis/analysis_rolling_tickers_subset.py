import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from parameters.parameters import Params


from training.trainer import Trainer
from util.dataframes_handling import load_all_pickles_in_folder
from plotting.Plotter import Plotter

if __name__ == "__main__":
    """
    Compute the performance filtering
    by ticker set
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions-folder",
        type=str,
        dest="predictions_folder",
        help="path to predictions_folder",
        required=True,
    )

    parser.add_argument(
        "--tickers-subset",
        type=str,
        dest="tickers_subset",
        help="path to tickers subset",
        required=True,
    )

    parser.add_argument("--t", type=int, dest="t", help="the time horizon")
    parser.set_defaults(t=5)

    parser.add_argument(
        "--quantile",
        type=float,
        dest="quantile",
        help="quantile used for the cross section definition",
    )
    parser.set_defaults(quantile=0)

    parser.add_argument(
        "--small-predictions-folder",
        type=str,
        dest="small_predictions_folder",
        help="folder to the small folder predictions",
    )
    args = parser.parse_args()

    plotter = Plotter()
    par = Params()

    par.data.t = args.t

    np.random.seed(par.seed)
    tf.random.set_seed(par.seed)

    trainer = Trainer(par)

    # load predictions
    dataset = load_all_pickles_in_folder(args.predictions_folder)
    small_dataset = load_all_pickles_in_folder(args.small_predictions_folder)

    tickers = pd.read_csv(args.tickers_subset)

    tickers = tickers.ticker.tolist()
    dataset = dataset[dataset.ticker.isin(tickers)]

    df, small_df, _ = trainer.load_benchmark(
        dataset,
        small_dataset,
        None,
        with_lag=True,
        is_small=True,
        quantile_threshold=args.quantile,
    )

    big_r2 = trainer.r2_f(df)
    small_r2 = trainer.r2_f(small_df)

    big_r2.to_csv("paper/res_paper/big_on_small_subset_r2.csv")
    small_r2.to_csv("paper/res_paper/small_subset_r2.csv")

    rolling = trainer.compute_rolling_r2(df)
    small_rolling = trainer.compute_rolling_r2(small_df)
    plotter.plot_multiple_lines(
        dfs=[rolling, small_rolling],
        x="index",
        y="Mean R^2",
        title="",
        xlabel="Year",
        ylabel="$R^2$",
        save_name="paper/res_paper/big_predicting_small_tickers_subset.png",
        labels=["DNN trained on full dataset", "DNN trained on small dataset"],
        colors=["black", "brown"],
        linestyles=["solid", "dotted"],
    )
