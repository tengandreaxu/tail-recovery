import gc
import os
import numpy as np
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

from parameters.parameters import ParamsData, Loss

import logging

logging.basicConfig(level=logging.INFO)


class DataReal:
    """
    A data object well structured for prediction tasks
    """

    def __init__(self, par: ParamsData):
        self.par = par
        self.y_train = None
        self.X_train = None
        self.y_test_val = None
        self.X_test_val = None
        self.y_test_future = None
        self.X_test_future = None

        self.future_ticker = None
        self.val_ticker = None
        self.train_ticker = None

        self.logger = logging.getLogger("DataReal")
        self.dir = os.path.join(self.par.data_path, "X_train")

    def load_all(
        self,
        save_columns=None,
        use_saved_columns=None,
        keep_all=False,
        lag=False,
        zero_column=None,
    ):
        """
        E.g. train on 2020, 2019, ... 2011, to predict 2010, 2009,2008
        :save_columns string, path to save the columns used for prediction
        :use_saved_columns list, list of columns saved and used to predict
        :keep_all
        """

        ticker = pd.read_pickle(os.path.join(self.dir, "train_ticker.p"))

        self.logger.info(f"train_ticker size={ticker.shape[0]}")

        # set the y to be the prediction horizon stated in parameters config
        if lag:
            rolling_std = "rolling_historical_std_zero_mean_window=" + str(
                self.par.prb_sig_est_window
            )

            shifted = (
                ticker.groupby("ticker")[["return_t=" + str(self.par.t)]]
                .shift(-1)
                .reset_index()
            )
            shifted_historical = (
                ticker.groupby("ticker")[[rolling_std]].shift(-1).reset_index()
            )

            ticker[["return_t=" + str(self.par.t)]] = shifted[
                "return_t=" + str(self.par.t)
            ].values

        y = ticker[["return_t=" + str(self.par.t)]].values

        if keep_all:
            self.y_fd = y.copy()
        # 1 gain, 0 loss

        ########################
        # For Tail Risk task
        ########################
        if self.par.loss in [Loss.PARETO_ONLY_SIGMA, Loss.EX_GDP_SIGMA]:
            self.logger.info(
                "{:<15} {:<15}".format(
                    f"Loss={self.par.loss}",
                    f"sigma={self.par.prb_sig_dist}",
                )
            )

            sig = self.par.prb_sig_dist

            if not lag:
                qv = ticker[
                    [
                        "rolling_historical_std_zero_mean_window="
                        + str(self.par.prb_sig_est_window)
                    ]
                ].values * np.sqrt(self.par.t)
            else:
                qv = shifted_historical[[rolling_std]].values * np.sqrt(self.par.t)

            # Threshold Exceedance cut
            if sig > 0.0:

                ind = y >= qv * sig
            else:
                ind = y <= qv * sig
            self.logger.info(f"Mean indicator: {ind.mean()}")
            self.logger.info(f"Data Points: {ind.sum()}")
            self.logger.info(f"Using lag={lag}")

            y[~ind] = np.nan

            # From Extreme value theory:
            # This is the excess distribution
            # Let u = qv * sig
            # Let Y the excess return random variable
            # Pr( Y - u <= y | y > u )
            y = np.abs(y) - np.abs(qv * sig)
            y = np.log(y + 0.000000001)

            if keep_all:
                self.y_fd = np.ones(self.y_fd.shape)
            # if self.par.name:
            #     quantile_name = os.path.join(
            #         self.par.res_dir, self.par.name, "qv.csv"
            #     )
            #     quantile = pd.DataFrame(qv, columns=["qv"])
            #     quantile.to_csv(quantile_name)

        if self.par.loss in [Loss.MSE_VOLA]:
            if self.par.t == 1:
                ticker["realized_std"] = ticker["return_t=1"] ** 2
            else:
                ticker["realized_std"] = ticker[
                    "rolling_realized_std_zero_mean_window=" + str(self.par.t)
                ] * np.sqrt(self.par.t)
            y = ticker[["realized_std"]].values
            logging.info(f"MSE VOLA data points: {ticker.shape[0]}")

        # pd.isna(df).mean()

        ##################
        # get the building blocks together
        ##################
        def drop_sparse_col(df_, use_saved_columns=use_saved_columns):
            """
            Keeps only those columns with less than 10% na
            """
            tr = 0.1
            col = pd.isna(df_).mean() <= tr

            # load same columns as stated
            if use_saved_columns:
                col.loc[~col.index.isin(use_saved_columns)] = False
            return df_.loc[:, col]

        df_columns = []
        df = []

        if self.par.ivs > -1:
            logging.info(f"Loading IVs")
            x_data = pd.read_pickle(os.path.join(self.dir, "ivs/X_train.p"))

            t = drop_sparse_col(x_data)

            # next we drop some moneyness or opt_type for some analysis
            if len(self.par.moneyness_only) > 0:
                c = [
                    x
                    for x in t.columns
                    if float(x.split("moneyness=")[1].split("_maturity")[0])
                    in self.par.moneyness_only
                ]
                t = t[c]
            if self.par.opt_type_only == -1:
                c = [x for x in t.columns if x[0] == str("p")]
                t = t[c]
            if self.par.opt_type_only == 1:
                c = [x for x in t.columns if x[0] == str("c")]
                t = t[c]

            ##############
            # Use Bid or Ask only IV
            #############
            if self.par.iv_type_only:
                c = [x for x in t.columns if self.par.iv_type_only in x.split("_")[0]]
                t = t[c]
            if zero_column and zero_column in t.columns:
                t[zero_column] = 0
            df.append(t.values)
            df_columns.append(list(t.columns))
            for i in range(1, self.par.ivs):

                t = drop_sparse_col(
                    pd.read_pickle(os.path.join(self.dir, f"ivs_t=-{str(i)}/X_train.p"))
                )
                df.append(t.values)
                df_columns.append([x + " lag " + str(i) for x in t.columns])
        if self.par.open_interests > -1:
            logging.info(f"Loading Open Interests")
            x_data = pd.read_pickle(os.path.join(self.dir, "open_interests/X_train.p"))

            t = drop_sparse_col(x_data)
            df.append(t.values)
            df_columns.append(list(t.columns))
            for i in range(1, self.par.open_interests):
                t = drop_sparse_col(
                    pd.read_pickle(
                        os.path.join(self.dir, f"open_interests_t=-{str(i)}/X_train.p")
                    ).iloc[:, :70]
                )
                df.append(t.values)
                df_columns.append([x + " lag " + str(i) for x in t.columns])
        if self.par.volumes > -1:
            logging.info("Loading Volumes")
            x_data = pd.read_pickle(os.path.join(self.dir, "volumes/X_train.p"))
            t = drop_sparse_col(x_data)
            df.append(t.values)
            df_columns.append(list(t.columns))
            for i in range(1, self.par.volumes):
                t = drop_sparse_col(
                    pd.read_pickle(
                        os.path.join(self.dir, f"/volumes=-{str(i)}/X_train.p")
                    ).iloc[:, :70]
                )
                df.append(t.values)
                df_columns.append([x + " lag " + str(i) for x in t.columns])
        if self.par.other > -1:
            temp = pd.read_pickle(os.path.join(self.dir, "other/X_train.p"))
            for t in [5, 20, 252]:
                temp["rolling_historical_std_zero_mean_window=" + str(t)] = ticker[
                    ["rolling_historical_std_zero_mean_window=" + str(t)]
                ]
            temp.iloc[:, :] = (temp.iloc[:, :] - temp.iloc[:, :].mean()) / temp.iloc[
                :, :
            ].std()
            df.append(temp.values)
            df_columns.append(list(temp.columns))
            del temp
        if self.par.realized > -1:
            temp = ticker[["rolling_realized_std_zero_mean_window=5"]] = ticker[
                ["rolling_realized_std_zero_mean_window=" + str(5)]
            ]
            for t in [20, 252]:
                temp["rolling_realized_std_zero_mean_window=" + str(t)] = ticker[
                    ["rolling_realized_std_zero_mean_window=" + str(t)]
                ]
            temp.iloc[:, :] = (temp.iloc[:, :] - temp.iloc[:, :].mean()) / temp.iloc[
                :, :
            ].std()
            df.append(temp.values)
            df_columns.append(list(temp.columns))
            del temp

        if self.par.moments > -1:
            logging.info(f"Loading Moments")
            C = []
            Temps = []

            for rolling_window in [5, 20, 252]:
                C.append(f"rolling_historical_skew_window={rolling_window}")
                t = (
                    ticker.groupby("ticker")["return_t=0"]
                    .rolling(rolling_window)
                    .skew()
                    .reset_index()
                    .rename(
                        columns={
                            "return_t=0": f"rolling_historical_skew_window={rolling_window}"
                        }
                    )
                )
                t["trade_date"] = pd.to_datetime(t["trade_date"])
                Temps.append(t)

                C.append(f"rolling_historical_kurtosis_window={rolling_window}")
                t = (
                    ticker.groupby("ticker")["return_t=0"]
                    .rolling(rolling_window)
                    .kurt()
                    .reset_index()
                    .rename(
                        columns={
                            "return_t=0": f"rolling_historical_kurtosis_window={rolling_window}"
                        }
                    )
                )
                t["trade_date"] = pd.to_datetime(t["trade_date"])
                if zero_column and zero_column in t.columns:
                    t[zero_column] = 0
                Temps.append(t)

            ticker = ticker.reset_index().rename(columns={"index": "trade_date"})
            ticker["trade_date"] = pd.to_datetime(ticker["trade_date"])

            for t in Temps:
                ticker = ticker.merge(t)

            del Temps

            ticker.index = ticker["trade_date"]
            ticker = ticker.drop(columns=["trade_date"])

            if use_saved_columns:
                C = [x for x in C if x in use_saved_columns]

            ###XXX TODO improve coding here
            temp = ticker.loc[:, C]
            temp.iloc[:, :] = (temp.iloc[:, :] - temp.iloc[:, :].mean()) / temp.iloc[
                :, :
            ].std()
            df.append(temp.values)
            df_columns.append(list(temp.columns))

            temp = ticker[["rolling_historical_std_zero_mean_window=" + str(5)]]
            for t in [20, 252]:
                temp["rolling_historical_std_zero_mean_window=" + str(t)] = ticker[
                    ["rolling_historical_std_zero_mean_window=" + str(t)]
                ]
            temp.iloc[:, :] = (temp.iloc[:, :] - temp.iloc[:, :].mean()) / temp.iloc[
                :, :
            ].std()
            if use_saved_columns:
                temp_columns = temp.columns.tolist()
                temp_columns = [x for x in temp_columns if x in use_saved_columns]
                temp = temp[temp_columns]
            df.append(temp.values)
            df_columns.append(list(temp.columns))

        df = np.concatenate(df, axis=1)
        all_col = []

        for l in df_columns:
            all_col += l
        self.columns = all_col
        if use_saved_columns:
            self.columns = [x for x in self.columns if x in use_saved_columns]

        if save_columns:
            predictors = pd.DataFrame(self.columns, columns=["predictor"])
            predictors.to_csv(
                os.path.join(save_columns, "used_columns.csv"), index=True
            )

            print("Saved the predictors as a csv file")
        if keep_all:
            self.y_fd, self.df_fd, self.ticker_fd = self.dataset_final_cleaning(
                self.y_fd, df.copy(), ticker.copy()
            )

        logging.info(f"Final Cleaning...")
        y, df, ticker = self.dataset_final_cleaning(y, df, ticker)
        ##################
        # finally we can split the sample
        ##################
        logging.info(f"Splitting data...")
        self._split_sample(ticker, df, y)

    def dataset_final_cleaning(
        self, y: np.ndarray, df: np.ndarray, ticker: pd.DataFrame
    ):
        #################
        # Still deleting NaNs
        #################
        ind_df = np.sum(np.isnan(df), axis=1) > 1
        ind_y = np.isnan(y[:, 0]) > 0

        ind = ind_y | ind_df

        y = y[~ind, :]
        df = df[~ind, :]
        df = np.nan_to_num(df)
        # clipping extreme values
        df = np.clip(df, -5.0, 5.0)
        ticker = ticker.loc[~ind, :]

        # next day vola are just squared returns
        ticker["rolling_realized_std_window=1"] = ticker["return_t=1"] ** 2
        ticker["rolling_realized_std_zero_mean_window=1"] = ticker["return_t=1"] ** 2
        ticker["rolling_historical_std_zero_mean_window=1"] = ticker["return_t=1"] ** 2

        ticker["realized_std"] = ticker[
            "rolling_realized_std_zero_mean_window=" + str(self.par.t)
        ] * np.sqrt(self.par.t)

        ticker["hist_pred_std"] = ticker[
            "rolling_historical_std_zero_mean_window=20"
        ] * np.sqrt(self.par.t)
        # (ticker['hist_pred_std'] - ticker['realized_std']).abs().mean()

        ticker["date"] = ticker.index
        ticker["date"] = pd.to_datetime(ticker["date"])
        ticker = ticker.reset_index(drop=True)
        ticker["year"] = ticker["date"].dt.year
        ticker["y"] = y
        # the month is defined as year * 100 + month
        # e.g.
        # input 2020-08-12
        # output 202008
        ticker["month"] = ticker["date"].dt.year * 100 + ticker["date"].dt.month
        return y, df, ticker

    def resplit_sample(self):
        ticker = self.train_ticker.append(self.future_ticker).reset_index(drop=True)
        df = np.concatenate([self.X_train, self.X_test_future], axis=0)
        y = np.concatenate([self.y_train, self.y_test_future], axis=0)
        # expanding window

        self._split_sample(ticker, df, y)

    def _split_sample(self, ticker, df, y):
        """
        Expanding rolling window

        Split into train, test and validation sets
        """
        # test sample size
        if self.par.test_cut_month > -1:
            ind = (
                ticker["month"]
                >= self.par.test_cut_year * 100 + self.par.test_cut_month
            )
        else:
            ind = ticker["year"] >= self.par.test_cut_year

        # train sample
        self.X_train = df[~ind, :]
        self.y_train = y[~ind, :]
        self.train_ticker = ticker.loc[~ind, :]

        # test, val sample
        df = df[ind, :]
        y = y[ind, :]
        ticker = ticker.loc[ind, :]
        ticker.loc[:, "val"] = False
        for year in ticker["year"].unique():
            l = ticker.loc[ticker["year"] == year, "ticker"].unique()
            tic = np.random.choice(l, size=int(np.round(len(l) * 0.05)))

            ticker.loc[
                (ticker["year"] == year) & ticker["ticker"].isin(tic), "val"
            ] = True

        # validation set
        ind = ticker["val"]
        self.X_test_val = df[ind, :]
        self.y_test_val = y[ind, :]
        self.val_ticker = ticker.loc[ind, :]

        # test set
        self.X_test_future = df[:, :]
        self.y_test_future = y[:, :]
        self.future_ticker = ticker.loc[:, :]

    def keep_only_future(self):
        """
        Keeps only future (test) data, while deletes
        train and val data
        """
        self.X_train = None
        self.y_train = None
        self.train_ticker = None
        self.X_test_val = None
        self.y_test_val = None
        self.val_ticker = None
        gc.collect()
