import os
import argparse

import pandas as pd
from util.dataframes_handling import load_all_pickles_in_folder


def compute_r2(df: pd.DataFrame):
    return 1 - ((df.y - df.pred) ** 2).sum() / ((df.y - df.y.mean()) ** 2).sum()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--covariate-results",
        type=str,
        dest="covariate_results",
        help="covariate results",
        required=True,
    )

    parser.add_argument(
        "--rolling-pred",
        type=str,
        dest="rolling_pred",
        help="path to rolling_pred",
        required=True,
    )

    args = parser.parse_args()

    covariates = os.listdir(args.covariate_results)

    standard_df = load_all_pickles_in_folder(args.rolling_pred)
    standard_r2 = compute_r2(standard_df)

    output = dict()
    output["all"] = standard_r2
    for covariate in covariates:

        path_to_covariate = f"{args.covariate_results}{covariate}/"
        covariate_df = load_all_pickles_in_folder(path_to_covariate)
        covariate_r2 = compute_r2(covariate_df)
        output[covariate] = covariate_r2
        print(
            f"covariate={covariate}, r2={round(covariate_r2,6)}, original_r2={round(standard_r2,6)}"
        )

    res = "/".join(args.covariate_results.split("/")[:-2])
    output = pd.DataFrame.from_dict(output, orient="index")
    output = output.rename(columns={0: "r2"})
    output.to_csv(f"{res}/covariates_r2.csv")
