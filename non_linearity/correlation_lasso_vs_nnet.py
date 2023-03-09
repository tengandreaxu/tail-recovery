import argparse
import os

import pandas as pd

from util.dataframes_handling import load_all_pickles_in_folder
from plotting.Plotter import Plotter

NON_LINEARITY_FOLDER = "paper/res_paper/non_linearity/"
os.makedirs(NON_LINEARITY_FOLDER, exist_ok=True)


def compute_correlation(
    df1: pd.DataFrame, df2: pd.DataFrame, column: str, message: str = None
):
    correlation = round(df1[column].corr(df2[column]), 3)

    print(f"{message} correlation: {correlation}")
    return correlation


if __name__ == "__main__":
    """
    Computes the correlation between

    corr(lasso_full_pos_pred, lasso_full_neg_pred)
    corr(nnet_full_pos_pred, nnet_full_neg_pred)
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--lasso-pos", dest="lasso_pos", type=str, required=True)

    parser.add_argument("--lasso-neg", dest="lasso_neg", type=str, required=True)

    parser.add_argument("--nnet-pos", dest="nnet_pos", type=str, required=True)

    parser.add_argument("--nnet-neg", dest="nnet_neg", type=str, required=True)

    args = parser.parse_args()

    lasso_neg = load_all_pickles_in_folder(args.lasso_neg)
    lasso_pos = load_all_pickles_in_folder(args.lasso_pos)

    ###################
    # Lasso
    ###################
    lasso_correlation = compute_correlation(lasso_pos, lasso_neg, "pred", "Lasso")
    plotter = Plotter()
    plotter.scatter_plot(
        x=lasso_pos.pred.values,
        y=lasso_neg.pred.values,
        title=f"Correlation: {str(lasso_correlation)}",
        ylabel="Negative Tail Predictions",
        xlabel="Positive Tail Predictions",
        grid=False,
        save_path=os.path.join(NON_LINEARITY_FOLDER, "lasso.png"),
        color="black",
        marker="+",
    )

    nnet_pos = load_all_pickles_in_folder(args.nnet_pos)
    nnet_neg = load_all_pickles_in_folder(args.nnet_neg)

    nnet_correlation = compute_correlation(nnet_pos, nnet_neg, "pred", "DNN")

    ###################
    # NNET
    ###################

    plotter.scatter_plot(
        x=nnet_pos.pred.values,
        y=nnet_neg.pred.values,
        title=f"Correlation: {str(nnet_correlation)}",
        ylabel="Negative Tail Predictions",
        xlabel="Positive Tail Predictions",
        grid=False,
        save_path=os.path.join(NON_LINEARITY_FOLDER, "nnet.png"),
        color="black",
        marker="+",
    )

    neg_neg = compute_correlation(nnet_neg, lasso_neg, "pred", "Neg vs Neg")

    plotter.scatter_plot(
        x=lasso_neg.pred.values,
        y=nnet_neg.pred.values,
        title=f"Correlation: {str(neg_neg)}",
        ylabel="Negative Tail DNN Predictions",
        xlabel="Negative Tail Lasso Predictions",
        grid=False,
        save_path=os.path.join(NON_LINEARITY_FOLDER, "neg_neg.png"),
        color="black",
        marker="+",
    )

    pos_pos = compute_correlation(nnet_pos, lasso_pos, "pred", "Pos vs Pos")

    plotter.scatter_plot(
        x=lasso_pos.pred.values,
        y=nnet_pos.pred.values,
        title=f"Correlation: {str(pos_pos)}",
        ylabel="Positive Tail DNN Predictions",
        xlabel="Positive Tail Lasso Predictions",
        grid=False,
        save_path=os.path.join(NON_LINEARITY_FOLDER, "pos_pos.png"),
        color="black",
        marker="+",
    )
