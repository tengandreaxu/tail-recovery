import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stats.distributions import exponential_gpd_mle, exponential_gpd_distribution
from util.dataframes_handling import load_all_pickles_in_folder


def plot_exponential_generalized_pareto_fit(df: pd.DataFrame, u, is_pos: bool):
    output_dir = "paper/res_paper/distributions/ex_gpd/"
    os.makedirs(output_dir, exist_ok=True)
    # ex_gpd
    sigma, xi = exponential_gpd_mle(df)
    X = np.arange(df.min(), df.max(), 0.01)
    y = [exponential_gpd_distribution(x, xi=xi, sigma=sigma) for x in X]

    plt.hist(df, bins=100, density=True, alpha=0.5, color="black")
    plt.xlabel("$y_{i,t}$")
    plt.ylabel("Density")
    plt.plot(X, y, color="black", label="exGPD")
    plt.tight_layout()
    plt.legend()
    xi = round(xi, 3)
    sigma = round(sigma, 3)
    plt.savefig(
        f"{output_dir}ex_gpd_comparison_sigma={sigma}_xi={xi}_is_pos={is_pos}.png"
    )
    plt.close()


if __name__ == "__main__":
    """
    We want to fit two exGPD, one for the negative tail and
    the other one for the positive tail.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full-dataset-prediction",
        "-fp",
        dest="full_predictions",
        type=str,
        help="path to the full dataset predictions",
    )
    parser.add_argument(
        "--is-pos",
        dest="is_pos",
        action="store_true",
        help="either is the positive tail or not",
    )

    args = parser.parse_args()
    sigma = -2.0
    if args.is_pos:
        sigma = 2.0

    df = load_all_pickles_in_folder(args.full_predictions)

    u = sigma * df["rolling_historical_std_zero_mean_window=252"].values
    df = df["y"].values
    plot_exponential_generalized_pareto_fit(df, u, args.is_pos)
