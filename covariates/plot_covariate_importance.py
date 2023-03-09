import os
import argparse

import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from plotting.Plotter import Plotter, pylab

MONEYNESS_2_NAME = {
    "_moneyness=0.0_": " ATM ",
    "_moneyness=1.0_": " ITM ",
    "_moneyness=-1.0_": " OTM ",
    "_moneyness=2.0_": " Deep ITM ",
    "_moneyness=-2.0_": " Deep OTM ",
}

MATURITY_BUCKETS = [0.0, 5.0, 15.0, 30.0, 60.0, 120.0, 250.0]


def make_pretty_bucket_names(bucket_names: list) -> list:

    bucket_names = [x.replace("_option_type=1", "") for x in bucket_names]
    bucket_names = [x.replace("_option_type=-1", "") for x in bucket_names]

    cleaned = []

    # keep the order
    for bucket_name in bucket_names:
        for key_ in MONEYNESS_2_NAME.keys():
            if key_ in bucket_name:
                cleaned.append(bucket_name.replace(key_, MONEYNESS_2_NAME[key_]))
            # it's a historical features
            elif bucket_name.startswith("rolling"):
                cleaned.append(bucket_name.replace("_", " ").replace("=", ": "))
                break

    cleaned = [x.replace("maturity_bucket", r"$\tau$") for x in cleaned]
    return cleaned


def plot_covariates(covariates: pd.DataFrame, file_name: str, full: bool):
    covariates = covariates.sort_values("r2", ascending=False)
    if not full:
        covariates = covariates.head(20)
        min_x = 0
    else:
        min_x = covariates.r2.min()
        params = {"ytick.labelsize": 4, "xtick.labelsize": 4}

        pylab.rcParams.update(params)
    fig, ax = plt.subplots()

    y_pos = np.arange(0, covariates.shape[0], 1)

    covariates.r2 = covariates.r2 / la.norm(covariates.r2, 1)
    xticks = np.arange(min_x, 0.15, step=0.02)
    ax.set_xticks(xticks)
    plt.grid(axis="x", linestyle="dashed")
    ax.barh(y_pos, covariates.r2.values, color="black")
    ax.set_yticks(y_pos)

    pretty_bucket_names = make_pretty_bucket_names(covariates.index.tolist())
    ax.set_yticklabels(pretty_bucket_names)
    ax.set_xlabel("Importance")
    ax.invert_yaxis()

    plt.tight_layout()
    if not full:
        fig.savefig(file_name)
    else:
        fig.savefig(file_name, dpi=300, bbox_inches="tight")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--covariate-r2",
        type=str,
        dest="covariate_r2",
        help="covariate r2 results",
        required=True,
    )

    parser.add_argument(
        "--full",
        action="store_true",
        dest="full",
        help="either plot all covariates or not",
        required=False,
    )

    parser.add_argument("--positive", action="store_true", dest="positive")
    parser.set_defaults(positive=False)
    args = parser.parse_args()

    covariates = pd.read_csv(args.covariate_r2)
    covariates = covariates.rename(columns={"Unnamed: 0": "covariate"})

    covariates.index = covariates.covariate
    covariates.pop("covariate")
    r2 = covariates.loc["all"].r2
    covariates = covariates.drop("all")
    covariates.r2 = r2 - covariates.r2
    output_folder = "paper/res_paper/covariates"
    os.makedirs(output_folder, exist_ok=True)
    file_name = os.path.join(
        output_folder, f"covariate_plot_is_pos={args.positive}.png"
    )
    if args.full:
        file_name = os.path.join(
            output_folder, f"covariate_plot_full_is_pos={args.positive}.pdf"
        )

    plot_covariates(covariates, file_name, args.full)
