import argparse
import os

import tensorflow as tf
import numpy as np
import pandas as pd
from parameters.parameters import ParamsData, Loss, SEED
from training.data import DataReal
import logging
from stats.distributions import exponential_gpd_mle, exponential_gpd_distribution
from plotting.Plotter import Plotter

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    """It computes the E[y|S] MLE, our benchmark for cross-sectional
    and time-series R^2"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--loss", dest="loss", type=int, help="the loss function under consideration"
    )
    parser.set_defaults(loss=Loss.EX_GDP_SIGMA)

    parser.add_argument(
        "--data-path", dest="data_path", type=str, help="path to the dataset"
    )
    parser.set_defaults(data_path=os.environ["GRIDS_FULL"])

    parser.add_argument(
        "--is-small",
        dest="is_small",
        action="store_true",
        help="either we are using the small dataset or not",
    )
    parser.set_defaults(is_small=False)
    args = parser.parse_args()

    bench_folder = os.environ["BENCH_DATA"]
    plotter = Plotter()

    dataset_parameters = ParamsData()
    dataset_parameters.ivs = 1
    dataset_parameters.volumes = -1
    dataset_parameters.open_interests = -1
    dataset_parameters.other = -1
    dataset_parameters.data_path = args.data_path
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    for t_dist in [1, 5]:
        for sigma in [-2, 2]:

            pos = "pos" if sigma == 2 else "neg"
            print("#" * 50)
            print("Start", t_dist)
            print("#" * 50)

            dir_ = os.path.join(
                bench_folder, f"exGDP_bench_lag_{pos}_{args.is_small}", str(t_dist)
            )

            os.makedirs(dir_, exist_ok=True)
            dataset_parameters.t = t_dist
            dataset_parameters.loss = args.loss
            dataset_parameters.prb_sig_dist = sigma

            data = DataReal(dataset_parameters)
            data.load_all(lag=True)

            # **************
            # we get all the observations in the tail
            # **************

            y_all = np.concatenate([data.y_train, data.y_test_future])
            ticker_all = data.train_ticker.append(data.future_ticker)
            ticker_all = ticker_all.reset_index(drop=True).reset_index()
            del data

            ##################
            # log likelihood in sample
            ##################
            sigma, xi = exponential_gpd_mle(y_all)
            final_par = pd.DataFrame(data={"sigma": sigma, "xi": xi}, index=["all"])

            X = np.arange(np.min(y_all), np.max(y_all), 0.01)
            y = [exponential_gpd_distribution(x, xi, sigma) for x in X]

            plotter.plot_histogram_with_curve(
                hist=y_all,
                curve=[X, y],
                bins=100,
                density=True,
                file_name=os.path.join(dir_, "y_in_sample.pdf"),
            )

            # compute the ll

            r = exponential_gpd_distribution(y_all, xi, sigma)
            ticker_all["ll_all"] = np.log(r)

            ##################
            # stock by stock
            ##################
            RES = [final_par]
            LL = []
            LL_ind = []
            for tic in ticker_all["ticker"].unique():
                ind = ticker_all["ticker"] == tic

                sigma, xi = exponential_gpd_mle(y_all[ind, 0])
                final_par = pd.DataFrame(data={"sigma": sigma, "xi": xi}, index=[tic])
                RES.append(final_par)
                # compute ll
                r = exponential_gpd_distribution(y_all[ind, :], xi, sigma)
                LL.append(r)
                LL_ind.append(ticker_all.index[ind])

            r = np.concatenate(LL)
            temp = pd.DataFrame(
                data={
                    "ll_firm": np.log(np.concatenate(LL).flatten()),
                    "index": np.concatenate(LL_ind).flatten(),
                }
            )
            ticker_all = ticker_all.merge(temp)

            df = pd.concat(RES, axis=0)
            df.to_csv(os.path.join(dir_, "bench.csv"))

            ##################
            # year by year
            ##################
            RES = []
            LL = []
            LL_ind = []
            for year in ticker_all["year"].unique():
                ind = ticker_all["year"] == year

                sigma, xi = exponential_gpd_mle(y_all[ind, 0])
                final_par = pd.DataFrame(data={"sigma": sigma, "xi": xi}, index=[year])
                RES.append(final_par)
                r = exponential_gpd_distribution(y_all[ind, :], xi, sigma)
                LL.append(r)
                LL_ind.append(ticker_all.index[ind])

            r = np.concatenate(LL)
            temp = pd.DataFrame(
                data={
                    "ll_year": np.log(np.concatenate(LL).flatten()),
                    "index": np.concatenate(LL_ind).flatten(),
                }
            )
            ticker_all = ticker_all.merge(temp)

            ticker_all[["ll_all", "ll_firm", "ll_year"]] = ticker_all[
                ["ll_all", "ll_firm", "ll_year"]
            ].replace({np.inf: np.nan})
            t = ticker_all[["ll_all", "ll_firm", "ll_year", "year"]].dropna()
            t = t.groupby("year")[["ll_all", "ll_firm", "ll_year"]].mean()
            t.to_csv(os.path.join(dir_, "ll.csv"))

            df = pd.concat(RES, axis=0)

            df.to_csv(os.path.join(dir_, "bench_year.csv"))
