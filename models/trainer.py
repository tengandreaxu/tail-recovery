import os
import gc
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression
from datetime import datetime


from plotting.Plotter import Plotter
from parameters.parameters import Params, Loss
from training.data import DataReal
from models.NetworkModel import NetworkModel
from stats.distributions import (
    exponential_gpd_expected_value_negative,
    exponential_gpd_expected_value_positive,
)
import logging

logging.basicConfig(level=logging.INFO)

EM_MODEL = r"$E[y_{i,t}|\tilde{\xi}_{i,t}, \tilde{\sigma}_{i,t}]$"
PRETTY_LABELS = {
    "Mean R^2": "$R^2$",
    "Mean R^2, firm": "$R^2_{firm}$",
    "Mean R^2, year": "$R^2_{year}$",
    "Mean R^2, rolling": "$R^2_{rolling}$",
}

PRETTY_MONEYNESS_COLUMNS = {
    "call_moneyness=-1.0": "Call OTM",
    "call_moneyness=-2.0": "Call Deep OTM",
    "call_moneyness=0.0": "Call ATM",
    "call_moneyness=1.0": "Call ITM",
    "call_moneyness=2.0": "Call Deep ITM",
    "put_moneyness=-1.0": "Put OTM",
    "put_moneyness=-2.0": "Put Deep OTM",
    "put_moneyness=0.0": "Put ATM",
    "put_moneyness=1.0": "Put ITM",
    "put_moneyness=2.0": "Put Deep ITM",
    "other_historical": "Other Historical",
}


class Trainer:
    def __init__(self, par: Params):
        self.par = par
        self.data = DataReal(par.data)

        self.model = NetworkModel(par.model, self.par.name)
        self.logger = logging.getLogger("Trainer")

    def update_par(self, par: Params):
        self.par = par
        self.data.par = par.data
        self.model.par = par.model

    def load_all_data(
        self,
        save_columns=None,
        use_saved_columns=None,
        keep_all=False,
        lag=False,
        zero_column=None,
    ):
        self.data.load_all(
            save_columns=save_columns,
            use_saved_columns=use_saved_columns,
            keep_all=keep_all,
            lag=lag,
            zero_column=zero_column,
        )

    def train_model(self):
        self.model.train(self.data.X_train, self.data.y_train, data=self.data)
        self.model.load(n=self.par.name)

    def save_par_in_res(self):
        self.par.save(self.model.res_dir + "/")

    def load_model(self):
        self.model.load(self.model.par.name)

    def get_pred(self, X, final):

        if self.par.model.loss in [Loss.EX_GDP_SIGMA]:
            df = self.model.x_to_human_input(X)
            sigma = df[["sigma"]].values
            xi = df[["xi"]].values

            # need values, otherwise
            # it will map by index
            final["xi_model"] = df["xi"].values
            final["sigma_model"] = df["sigma"].values

            # expectation
            m = (
                np.log(sigma / xi)
                + tf.math.digamma(1.0).numpy()
                - tf.math.digamma(1.0 / xi).numpy()
            )
            # std
            s = tf.math.polygamma(a=1.0, x=1.0) + tf.math.polygamma(a=1.0, x=1.0 / xi)

            final["s_pred"] = s.numpy() ** (1 / 2)
            final["pred"] = m

            # quantile
            q_inc = 0.05
            Q = np.arange(q_inc, 1.0, q_inc)
            for q in Q:
                final["q" + str(int(q * 100))] = np.log(
                    sigma / xi * (np.exp(-xi * np.log(1 - q)) - 1)
                )

            final["q100"] = 1000000.0
            final["q0"] = -1000000.0

        if self.par.model.loss in [Loss.MAE, Loss.MSE, Loss.R2]:
            df = self.model.model.predict(X)
            final["s_pred"] = 0.07
            final["pred"] = df

        if self.par.model.loss in [Loss.MSE_VOLA]:
            df = self.model.model.predict(X)
            final["s_pred"] = df
            final["pred"] = 0.0

        if self.par.model.loss in [Loss.SIGN]:
            df = (self.model.model.predict(X) > 0.5) * 0.02 - 0.01
            final["s_pred"] = 0.07
            final["pred"] = df

        if self.par.model.loss in [Loss.PARETO_ONLY_QUANTILE, Loss.PARETO_ONLY_SIGMA]:
            df = self.model.x_to_human_input(X)
            for c in df.columns:
                df[c] = df[c].astype(np.float32)

            d = tfp.distributions.GeneralizedPareto(
                loc=0.0, scale=df["sigma"], concentration=df["xi"]
            )
            final.loc[:, "s_pred"] = 0.01
            final.loc[:, "s_pred"] = d.stddev().numpy()
            final["s_pred"] = final["s_pred"].fillna(method="ffill")
            final["s_pred"] = final["s_pred"].fillna(method="bfill")
            final["s_pred"] = final["s_pred"].fillna(final["s_pred"].mean())
            final["s_pred"] = final["s_pred"].fillna(0.01)
            final["pred"] = d.mean()

            ##################
            # add the quantile approache
            #################
            q_inc = 0.05
            Q = np.arange(q_inc, 1.0, q_inc)
            for q in Q:
                final["q" + str(int(q * 100))] = d.quantile(q)

            final["q100"] = 1000000.0
            final["q0"] = -1000000.0

        if self.par.model.loss in [
            Loss.LOG_LIKE_PARETO,
            Loss.PARETO_FIX_U,
            Loss.PARETO_FIX_SIG_DIST,
            Loss.PARETO_SIG_DIST,
        ]:
            df = self.model.x_to_human_input(X)
            for c in df.columns:
                df[c] = df[c].astype(np.float32)

            d_n = tfp.distributions.Normal(loc=df["mu_n"], scale=df["sig_n"])
            d_n_trunc = tfp.distributions.TruncatedNormal(
                loc=df["mu_n"], scale=df["sig_n"], low=df["u_d"], high=df["u_u"]
            )
            d_p_u = tfp.distributions.GeneralizedPareto(
                loc=df["u_u"], scale=df["sig_p_u"], concentration=df["xi_u"]
            )
            d_p_d = tfp.distributions.GeneralizedPareto(
                loc=df["u_d"], scale=df["sig_p_d"], concentration=df["xi_d"]
            )

            p_l = d_n.cdf(df["u_d"]).numpy()
            p_u = 1 - d_n.cdf(df["u_u"]).numpy()

            m_p_low = 2 * df["u_d"].values - d_p_d.mean()
            m_p_high = d_p_u.mean()
            m_norm = d_n_trunc.mean()

            m = p_l * m_p_low + (1 - p_u - p_l) * m_norm + p_u * m_p_high

            s_p_low = d_p_d.variance().numpy()
            s_p_high = d_p_u.variance().numpy()
            s_norm = d_n_trunc.variance().numpy()

            s = (
                p_l * (s_p_low + m_p_low ** 2)
                + p_u * (s_p_high + m_p_high ** 2)
                + (1 - p_u - p_l) * (s_norm + m_norm ** 2)
            ) - m ** 2

            final["s_pred"] = s ** (1 / 2)  # * np.sqrt(1 / 5) * 5
            final["pred"] = m

            ##################
            # add the quantile approache
            #################
            q_inc = 0.05
            Q = np.arange(q_inc, 1.0, q_inc)
            for q in Q:
                t = 1 - (q / p_l)
                # t = (1 - (q + q_inc)) / p_l
                q_low = d_p_d.quantile(t).numpy()
                q_low = 2 * df["u_d"] - q_low
                q_norm = d_n.quantile(q).numpy()
                t = (q - (1 - p_u)) / p_u
                q_high = d_p_u.quantile(t).numpy()

                ind_low = q < p_l
                ind_high = q > (1 - p_u)
                q_norm[ind_low] = q_low[ind_low]
                q_norm[ind_high] = q_high[ind_high]
                final["q" + str(int(q * 100))] = q_norm

            final["q100"] = 1000000.0
            final["q0"] = -1000000.0

        if self.par.model.loss in [Loss.LOG_LIKE_RESTRICTED, Loss.LOG_LIKE]:
            MU, SIG, PI = self.model.x_to_human_input(X)
            ind = np.argsort(-MU, axis=1)

            pi_sort = np.take_along_axis(PI, ind, axis=1)
            mu_sort = np.take_along_axis(MU, ind, axis=1)
            sig_sort = np.take_along_axis(SIG, ind, axis=1)

            df = [
                pd.DataFrame(
                    pi_sort, columns=["pi_" + str(i) for i in range(PI.shape[1])]
                ),
                pd.DataFrame(
                    mu_sort, columns=["mu_" + str(i) for i in range(PI.shape[1])]
                ),
                pd.DataFrame(
                    sig_sort, columns=["sigma_" + str(i) for i in range(PI.shape[1])]
                ),
            ]
            df = pd.concat(df, 1)

            r_pred = 0
            sig_pred = 0
            for i in range(int(df.shape[1] / 3)):
                r_pred += df.loc[:, "mu_" + str(i)] * df.loc[:, "pi_" + str(i)]
                sig_pred += df.loc[:, "pi_" + str(i)] * (
                    df.loc[:, "sigma_" + str(i)] ** 2 + df.loc[:, "mu_" + str(i)] ** 2
                )
            sig_pred = sig_pred - r_pred ** 2

            final["s_pred"] = sig_pred.values ** (1 / 2)  # * np.sqrt(1 / 5) * 5
            final["pred"] = r_pred.values

            nb_normal = MU.shape[1]
            dist = []
            for i in range(nb_normal):
                dist.append(
                    tfp.distributions.Normal(df["mu_" + str(i)], df["sigma_" + str(i)])
                )

            # need to find x, such that cdf = q
            cdf = 0
            x = -0.08
            for i in range(nb_normal):
                cdf += dist[i].cdf(x).numpy() * df["pi_" + str(i)].values
            q10 = 0.01
            q15 = 0.02

            c = ["pi_" + str(i) for i in range(nb_normal)]
            cat = tfp.distributions.Categorical(df[c])
            d = tfp.distributions.MixtureSameFamily(
                mixture_distribution=cat, components_distribution=dist
            )

        return final

    def get_non_zero_columns_only_from_lasso_coef(
        self, coefs: np.ndarray
    ) -> pd.DataFrame:
        """
        Given a coefs list gives in output a DataFrame with only those
        columns with non zero coef
        """
        df = pd.DataFrame(data={"columns": self.data.columns, "coef": coefs})
        df = df[df.coef != 0]
        return df

    def model_predict(self, model, dir_: str, month: int, indexing=None):
        """
        statsmodel has inverted X,y
        """

        d = self.model.res_dir + dir_
        os.makedirs(d, exist_ok=True)

        final = self.data.future_ticker
        ind = final["month"] == month
        final = final.loc[ind, :]
        X = self.data.X_test_future[ind, :]

        # use only a subset of the predictors
        if indexing:
            X = [np.take(x, indexing) for x in X]

        Y = self.data.y_test_future[ind, :]
        final["realized_std"] = pd.to_numeric(final["realized_std"])
        final["pred"] = model.predict(X)
        final.to_pickle(d + str(month) + ".p")

    def lasso_full_dataset_month_pred(self, model, month: int, overwrite_dir=None):
        """
        Full dataset pred on nnet expanding window
        :param sub_dir:
        """

        output_dir = "/rolling_lasso_full_pred/"
        if overwrite_dir:
            output_dir = overwrite_dir
        d = self.model.res_dir + output_dir
        os.makedirs(d, exist_ok=True)

        final = self.data.ticker_fd
        ind = final["month"] == month

        final = final.loc[ind, :]
        X = self.data.df_fd[ind, :]
        Y = self.data.y_fd[ind, :]
        final["realized_std"] = pd.to_numeric(final["realized_std"])
        final["pred"] = model.predict(X)
        final.to_pickle(d + str(month) + ".p")

    def lasso_month_by_month(self, save_ols_parameters=False):
        """
        Trains a Lasso model
        given an array of alphas - the penalty coef
        """
        inc = 0.005
        ALPHA = np.arange(0.0, inc * 5, inc)
        M = []
        S = []
        COEF = []
        for alpha in ALPHA:
            m = Lasso(alpha, fit_intercept=True, max_iter=10 ** 4)
            m.fit(self.data.X_train, self.data.y_train)
            s = m.score(self.data.X_test_val, self.data.y_test_val)
            print(
                "{:<15} {:<15} {:<15}".format(
                    f"alpha={alpha}", f"score={s}", f"non_zero_coef={np.sum(m.coef_>0)}"
                )
            )

            S.append(s)
            M.append(m)
            COEF.append(np.sum(m.coef_ > 0))

        i = np.argmax(S)
        m = M[i]
        print("select", ALPHA[i])
        month = self.par.data.test_cut_year * 100 + self.par.data.test_cut_month
        if save_ols_parameters:
            ## used predictors from best Lasso
            used_predictors = self.get_non_zero_columns_only_from_lasso_coef(m.coef_)
            indexing = used_predictors.index.tolist()
            X_train_ols = [np.take(x, indexing) for x in self.data.X_train]
            ols = sm.OLS(self.data.y_train.flatten(), X_train_ols)
            ols_model = ols.fit()
            # for each predictor used in Lasso
            # we now have they p-value, t-stats
            coef_table = ols_model.summary().tables[1]
            html_table = coef_table.as_html()
            final_table = pd.read_html(html_table, header=0, index_col=0)[0]
            final_table["predictor_name"] = used_predictors["columns"].tolist()
            d = self.model.res_dir + "/rolling_ols_parameters/"
            os.makedirs(d, exist_ok=True)
            final_table["indexing"] = indexing
            final_table.to_pickle(d + str(month) + ".p")

            self.model_predict(
                ols_model, "/rolling_ols_with_lasso_beta/", month, indexing=indexing
            )

        m.score(self.data.X_test_future, self.data.y_test_future)
        lasso_betas = pd.DataFrame(data={"columns": self.data.columns, "coef": m.coef_})

        lasso_betas_folder = os.path.join(self.model.res_dir, "rolling_lasso_betas")
        os.makedirs(lasso_betas_folder, exist_ok=True)
        lasso_betas.to_csv(os.path.join(lasso_betas_folder, str(month)), index=False)

        self.model_predict(m, "/rolling_lasso/", month)
        self.lasso_full_dataset_month_pred(m, month)
        d = self.model.res_dir + "/rolling_lasso_parameters/"
        if not os.path.exists(d):
            os.makedirs(d)

        df = pd.DataFrame(data={"alpha": ALPHA, "score": S, "nb_coef": COEF})
        df.to_pickle(d + str(month) + ".p")

    def ols_with_lasso_parameter_month_by_month(self, year):
        lasso_parameters_folder = self.mode.res_dir + "/rolling_ols_parameters/"
        lasso_parameters = pd.read_pickle(lasso_parameters_folder + year + ".p")
        # m.fit(self.data.X_train, self.data.y_train)
        # m.score(self.data.X_test_future, self.data.y_test_future)

        d = self.model.res_dir + "/rolling_ols/"
        if not os.path.exists(d):
            os.makedirs(d)

        month = self.par.data.test_cut_year * 100 + self.par.data.test_cut_month
        final = self.data.future_ticker
        ind = final["month"] == month
        final = final.loc[ind, :]
        X = self.data.X_test_future[ind, :]
        Y = self.data.y_test_future[ind, :]
        final["realized_std"] = pd.to_numeric(final["realized_std"])
        final["pred"] = m.predict(X)
        final.to_pickle(d + str(month) + ".p")

    def ols_month_by_month(self):

        m = LinearRegression(fit_intercept=True)
        m.fit(self.data.X_train, self.data.y_train)

        m.score(self.data.X_test_future, self.data.y_test_future)

        d = self.model.res_dir + "/rolling_ols/"

        if not os.path.exists(d):
            os.makedirs(d)

        month = self.par.data.test_cut_year * 100 + self.par.data.test_cut_month

        # subsample of tail events
        final = self.data.future_ticker
        ind = final["month"] == month
        final = final.loc[ind, :]
        X = self.data.X_test_future[ind, :]
        Y = self.data.y_test_future[ind, :]
        final["realized_std"] = pd.to_numeric(final["realized_std"])
        final["pred"] = m.predict(X)
        final.to_pickle(d + str(month) + ".p")

        output_dir = os.path.join(self.model.res_dir, "rolling_ols_full_pred/")
        os.makedirs(output_dir, exist_ok=True)

        # full sample
        final = self.data.ticker_fd
        ind = final["month"] == month
        final = final.loc[ind, :]
        X = self.data.df_fd[ind, :]
        Y = self.data.y_fd[ind, :]
        final["realized_std"] = pd.to_numeric(final["realized_std"])
        final["pred"] = m.predict(X)
        final.to_pickle(os.path.join(output_dir, f"{str(month)}.p"))

    def exGDP_month_pred(self, overwrite_dir=None):
        """
        In this method we just predict first out of sample month of data (for rolling window training)
        :param sub_dir:
        """

        output_dir = "rolling_pred/"
        if overwrite_dir:
            output_dir = overwrite_dir

        d = os.path.join(self.model.res_dir, output_dir)

        os.makedirs(d, exist_ok=True)

        # locking month of interested only
        month = self.par.data.test_cut_year * 100 + self.par.data.test_cut_month
        final = self.data.future_ticker
        ind = final["month"] == month
        ind.mean()
        final = final.loc[ind, :]

        X = self.data.X_test_future[ind, :]
        Y = self.data.y_test_future[ind, :]

        final["realized_std"] = pd.to_numeric(final["realized_std"])
        # final[std_bench] = pd.to_numeric(final[std_bench])
        final = self.get_pred(X, final)
        final.to_pickle(d + str(month) + ".p")

    def full_dataset_month_pred(self, overwrite_dir=None):
        """
        Full dataset pred on nnet expanding window
        :param sub_dir:
        """

        output_dir = "/rolling_nnet_full_pred/"
        if overwrite_dir:
            output_dir = overwrite_dir
        d = self.model.res_dir + output_dir
        os.makedirs(d, exist_ok=True)

        # locking month of interested only
        month = self.par.data.test_cut_year * 100 + self.par.data.test_cut_month
        final = self.data.ticker_fd
        ind = final["month"] == month
        final = final.loc[ind, :]
        X = self.data.df_fd[ind, :]
        Y = self.data.y_fd[ind, :]
        final["realized_std"] = pd.to_numeric(final["realized_std"])
        final = self.get_pred(X, final)
        final.to_pickle(d + str(month) + ".p")

    def plot_average_significant_p_values(self):
        """
        Plots time-series showing the average number of significant
        p-values

        grouped by moneyness, option_type
        """
        dir_ = self.model.res_dir + "/rolling_ols_parameters/"
        files_ = os.listdir(dir_)

        df = pd.DataFrame()
        for file_ in files_:
            sub_df = pd.read_pickle(dir_ + file_)

            sub_df = sub_df.rename(columns={"P>|t|": "p_value"})
            sub_df["month"] = datetime.strptime(file_.replace(".p", ""), "%Y%m").date()
            df = pd.concat([df, sub_df])

        predictors = df.predictor_name.tolist()
        predictors = [x.split("_") for x in predictors]

        cleaned = []

        ## we are interested on (option_type, moneyness) only
        for predictor in predictors:
            # e.g. rolling historical std zero mean window 252
            call_or_put = "other"
            type_ = predictor[-1]
            if type_ == "type=1":
                call_or_put = "call"
            if type_ == "type=-1":
                call_or_put = "put"

            if type_ != "other":
                moneyness = predictor[1]
            else:
                moneyness = ""
            cleaned.append(call_or_put + "_" + moneyness)
        df["predictor"] = cleaned

        df["macro_predictor"] = df.predictor.str.split("_")
        df["macro_predictor"] = df["macro_predictor"].apply(
            lambda x: x[0] if x[0] != "other" else x[0] + "_" + x[1]
        )

        # relevant predictor
        df = df[df.p_value < 0.05]
        df = df.sort_values("month")
        to_plot = pd.DataFrame()
        to_plot_macro = pd.DataFrame()

        months = df["month"].unique().tolist()

        macro_plot = df.copy()
        plotter = Plotter()
        for i in range(12, len(months)):

            m_start = months[i - 12]
            m_end = months[i]
            sub = df[(df.month >= m_start) & (df.month <= m_end)]

            sub_macro = macro_plot[(df.month >= m_start) & (df.month <= m_end)]
            sub = (
                sub.groupby(["predictor", "month"])
                .agg("count")["p_value"]
                .reset_index()
            )
            sub_macro = (
                sub_macro.groupby(["macro_predictor", "month"])
                .agg("count")["p_value"]
                .reset_index()
            )

            sub = sub.groupby(["predictor"]).mean()
            sub_macro = sub_macro.groupby(["macro_predictor"]).mean()

            sub = pd.DataFrame(sub).T
            sub_macro = pd.DataFrame(sub_macro).T
            sub.index = [m_end]
            sub_macro.index = [m_end]
            to_plot = pd.concat([to_plot, sub])
            to_plot_macro = pd.concat([to_plot_macro, sub_macro])

        output_dir = self.model.res_dir + "/ols_params/"
        os.makedirs(output_dir, exist_ok=True)

        # big category plot
        plotter.plot_multiple_lines_from_columns(
            to_plot_macro,
            xlabel="Year",
            ylabel="Count",
            save_name="macro.png",
            output_dir=output_dir,
            title="",
        )

        #################################
        # In all plots we keep other-historical then do plots:
        # 1. put only (one line per moneyness)
        # 2. call only (on line per moneyness)
        # 3. for each moneyness on plot, one line per call/put
        #################################
        to_plot = to_plot.rename(columns=PRETTY_MONEYNESS_COLUMNS)
        for plot_type in ["Call", "Put", "moneyness"]:
            to_plot_columns = to_plot.columns.tolist()
            if plot_type != "moneyness":
                to_plot_columns = [
                    x
                    for x in to_plot_columns
                    if x.startswith(plot_type) or x == "Other Historical"
                ]
                plotter.plot_multiple_lines_from_columns(
                    to_plot[to_plot_columns],
                    xlabel="Year",
                    ylabel="Count",
                    save_name=f"{plot_type}_only.png",
                    output_dir=output_dir,
                    title="",
                )
            else:
                for moneyness in ["Deep ITM", "ITM", "ATM", "OTM", "Deep OTM"]:
                    to_plot_columns = to_plot.columns.tolist()
                    put = "Put " + moneyness
                    call = "Call " + moneyness
                    to_plot_columns = [
                        x
                        for x in to_plot_columns
                        if put == x or call == x or x == "Other Historical"
                    ]
                    plotter.plot_multiple_lines_from_columns(
                        to_plot[to_plot_columns],
                        xlabel="Year",
                        ylabel="Count",
                        save_name=f"{plot_type}_moneyness={moneyness}_only.png",
                        output_dir=output_dir,
                        title="",
                    )

    def get_pred_month_mean(self, df: pd.DataFrame, y_column: str, is_panel: bool):
        """
        df must contain month and pred column
        df is the result of our model predictions
        """
        if is_panel:
            output = df[["month", "ticker", y_column]].copy()

            output = output.groupby(["ticker", "month"]).agg("mean").reset_index()
            output.index = output.ticker + "_" + output.month.astype(str)
            output = output.sort_index()
            output.pop("ticker")
        else:
            output = df[["month", y_column]].copy()
            output = output.groupby("month").mean().sort_index()
            output["month"] = output.index
        return output

    def load_benchmark(
        self,
        final,
        lasso_df,
        ols_df,
        with_lag=False,
        is_pos=False,
        is_small=False,
    ):
        bench_root = os.environ["BENCH_DATA"]
        ########################
        # These folders are ALL with lags
        ########################

        if with_lag and not is_pos:
            bench_folder = os.path.join(bench_root, "exGDP_bench_lag_neg/")
        elif with_lag and is_pos:
            bench_folder = os.path.join(bench_root, "exGDP_bench_lag_pos/")

        if is_small:
            if with_lag and not is_pos:
                bench_folder = os.path.join(bench_root, "exGDP_bench_lag_neg_True/")
            elif with_lag and is_pos:
                bench_folder = os.path.join(bench_root, "exGDP_bench_lag_pos_True/")

        self.logger.info("=" * 47)
        self.logger.info(f"Loading From: {bench_folder}")
        self.logger.info("=" * 47)

        ###################
        # Stock by Stock
        ###################

        bench = pd.read_csv(
            f"{bench_folder}{str(self.par.data.t)}/bench.csv", index_col=0
        )

        sigma = bench.loc["all", "sigma"]
        xi = bench.loc["all", "xi"]
        # We consider the case where \xi > 0
        single_mean = exponential_gpd_expected_value_positive(xi=xi, sigma=sigma)
        self.single_mean_sample = final.y.mean()
        sigma = bench.drop(index="all").loc[:, "sigma"]
        xi = bench.drop(index="all").loc[:, "xi"]

        tic_mean = exponential_gpd_expected_value_positive(xi=xi, sigma=sigma)
        tic_mean_neg = exponential_gpd_expected_value_negative(xi=xi, sigma=sigma)

        tic_mean[pd.isna(tic_mean)] = tic_mean_neg[pd.isna(tic_mean)]
        tic_mean = tic_mean.reset_index()
        tic_mean.columns = ["ticker", "mean_ticker"]

        final = final.merge(tic_mean)
        if lasso_df is not None:
            lasso_df = lasso_df.merge(tic_mean)
        if ols_df is not None:
            ols_df = ols_df.merge(tic_mean)

        ######################
        # Year by Year
        #####################
        bench = (
            pd.read_csv(
                f"{bench_folder}" + str(self.par.data.t) + "/bench_year.csv",
                index_col=0,
            )
            .reset_index()
            .rename(columns={"index": "year"})
        )
        sigma = bench.loc[:, "sigma"]
        xi = bench.loc[:, "xi"]
        tic_mean = exponential_gpd_expected_value_positive(xi=xi, sigma=sigma)
        tic_mean_neg = exponential_gpd_expected_value_negative(xi=xi, sigma=sigma)
        tic_mean[pd.isna(tic_mean)] = tic_mean_neg[pd.isna(tic_mean)]
        bench["mean_yy"] = tic_mean.values
        bench = bench.drop(columns=["xi", "sigma"])

        bench["year"] = bench["year"].astype(int)

        final = final.merge(bench)

        if lasso_df is not None:
            lasso_df = lasso_df.merge(bench)
        if ols_df is not None:
            ols_df = ols_df.merge(bench)

        self.single_mean = single_mean
        return final, lasso_df, ols_df

    def compute_rolling_r2(
        self, df: pd.DataFrame, use_normal_r2: bool = False
    ) -> pd.DataFrame:
        MONTH = df["month"].sort_values().unique()
        rolling_month = []
        for i in range(12, len(MONTH)):
            m_start = MONTH[i - 12]
            m_end = MONTH[i]
            ind = (df["month"] >= m_start) & ((df["month"] <= m_end))
            r = self.r2_f(df.loc[ind, :], use_normal_r2)
            r.name = df.loc[ind, "date"].min()
            rolling_month.append(r)
        rolling_month = pd.DataFrame(rolling_month)
        return rolling_month

    def r2_f(self, df_, use_normal_r2=False):

        if use_normal_r2:
            m = 1 - (
                ((df_["y"] - df_["pred"]) ** 2).sum()
                / ((df_["y"] - self.single_mean_sample) ** 2).sum()
            )

        else:
            m = 1 - (
                ((df_["y"] - df_["pred"]) ** 2).sum()
                / ((df_["y"] - self.single_mean) ** 2).sum()
            )
        m_hard = 1 - (
            ((df_["y"] - df_["pred"]) ** 2).sum()
            / ((df_["y"] - df_["mean_ticker"]) ** 2).sum()
        )
        m_year = 1 - (
            ((df_["y"] - df_["pred"]) ** 2).sum()
            / ((df_["y"] - df_["mean_yy"]) ** 2).sum()
        )

        data = {
            "Mean R^2": m,
            "Mean R^2, firm": m_hard,
            "Mean R^2, year": m_year,
        }
        return pd.Series(data)

    def analyze_exGDP(
        self,
        final=None,
        lasso_df=None,
        ols_df=None,
        with_lag=False,
        is_pos=False,
        is_small=False,
        plot_p_values=True,
        use_normal_r2=False,
    ):
        self.logger.info(f"will use_normal_r2={use_normal_r2}")

        # the OLS p value analysis
        if lasso_df is not None and ols_df is not None and plot_p_values:
            self.plot_average_significant_p_values()

        final["realized_std"] = pd.to_numeric(final["realized_std"])
        # create scatter plot analysis

        self.produce_scatter_plots(final)

        ##################
        # R^2
        ##################
        d = self.model.res_dir + "/r2/"
        if not os.path.exists(d):
            os.makedirs(d)

        # load the benchmark
        final, lasso_df, ols_df = self.load_benchmark(
            final,
            lasso_df,
            ols_df,
            with_lag=with_lag,
            is_pos=is_pos,
            is_small=is_small,
        )

        r2 = self.r2_f(final)
        r2.to_csv(os.path.join(d, "rtot.csv"))
        r_y = final.groupby("year").apply(self.r2_f)
        r_y.to_csv(os.path.join(d, "r_year.csv"))
        print("######## r2 score ####")
        print(r2)
        print(r_y)
        print("######################")
        final.groupby("ticker").apply(self.r2_f).to_csv(d + "r_ticker.csv")

        rolling_month = self.compute_rolling_r2(final, use_normal_r2)
        nnet_r2 = self.r2_f(final)
        nnet_r2.to_csv(f"{d}nnet_r2.csv")
        for c in rolling_month.columns:
            rolling_month.loc[:, c].plot(color="k", label="DNN", rot=0)
            plt.grid()
            plt.title(c.replace("R^2", "$R^2$"))
            plt.xlabel("Year")
            plt.ylabel(PRETTY_LABELS[c])
            plt.tight_layout()
            plt.savefig(
                d + "nnet_rolling_r2" + c.replace("^", "").replace(" ", "_") + ".png"
            )
            plt.close()

        if lasso_df is not None:
            rolling_month_lasso = self.compute_rolling_r2(lasso_df, use_normal_r2)
            lasso_r2 = self.r2_f(lasso_df)
            lasso_r2.to_csv(f"{d}lasso_r2.csv")
            rolling_month_lasso.to_pickle(d + "rolling_month_lasso.p")
            for c in rolling_month.columns:
                rolling_month.loc[:, c].plot(color="black", label="DNN", rot=0)
                rolling_month_lasso.loc[:, c].plot(
                    color="brown", linestyle=":", label="Lasso", rot=0
                )
                plt.legend()
                plt.grid()

                plt.xlabel("Year")

                plt.ylabel(PRETTY_LABELS[c])

                plt.tight_layout()
                plt.ylim([-0.33, 0.3])
                plt.savefig(
                    d
                    + "lasso_vs_nnet_rolling_"
                    + c.replace("^", "").replace(" ", "_")
                    + ".png"
                )
                plt.show()
                plt.close()

        if ols_df is not None:
            rolling_month_ols = self.compute_rolling_r2(ols_df, use_normal_r2)
            ols_r2 = self.r2_f(ols_df)
            ols_r2.to_csv(f"{d}ols_r2.csv")

            # save rolling_months
            rolling_month.to_pickle(d + "rolling_month_nnet.p")
            rolling_month_ols.to_pickle(d + "rolling_month_ols.p")

            for c in rolling_month.columns:
                rolling_month.loc[:, c].plot(color="k", label="DNN", rot=0)
                rolling_month_ols.loc[:, c].plot(
                    color="blue", linestyle="--", label="OLS", rot=0
                )
                plt.legend()
                plt.grid()
                plt.xlabel("Year")
                plt.ylabel(PRETTY_LABELS[c])
                plt.tight_layout()
                plt.savefig(
                    d
                    + "ols_vs_nnet_rolling_"
                    + c.replace("^", "").replace(" ", "_")
                    + ".png"
                )
                plt.show()
                plt.close()
            for c in rolling_month.columns:
                rolling_month_ols.loc[:, c].plot(
                    color="black", linestyle="--", label="OLS", rot=0
                )
                plt.legend()
                plt.grid()
                plt.xlabel("Year")
                plt.ylabel(PRETTY_LABELS[c])
                plt.ylim(-0.3, 0.22)
                plt.tight_layout()
                plt.savefig(
                    d
                    + "ols_vs_em_rolling_"
                    + c.replace("^", "").replace(" ", "_")
                    + ".png"
                )
                plt.show()
                plt.close()

        if ols_df is not None and lasso_df is not None:
            for c in rolling_month.columns:
                rolling_month.loc[:, c].plot(color="k", label="DNN", rot=0)
                rolling_month_lasso.loc[:, c].plot(
                    color="grey", linestyle=":", label="Lasso", rot=0
                )
                rolling_month_ols.loc[:, c].plot(
                    color="blue", linestyle="--", label="OLS", rot=0
                )
                plt.legend()
                plt.grid()
                plt.xlabel("Year")
                plt.ylabel(PRETTY_LABELS[c])
                plt.tight_layout()

                plt.savefig(
                    d
                    + "ols_vs_lasso_vs_nnet_rolling_"
                    + c.replace("^", "").replace(" ", "_")
                    + ".png"
                )
                plt.show()
                plt.close()

    def plot_predicted_parameters(self, final: pd.DataFrame):

        ##################
        # describe predicted parameters
        ##################

        try:
            d = self.model.res_dir + "/pred_params/"
            if not os.path.exists(d):
                os.makedirs(d)

            final.groupby("date")[["xi_model"]].mean().rolling(252).mean().plot(
                legend=False
            )
            plt.ylabel(r"$\xi$")
            plt.xlabel("Date")
            plt.savefig(d + "rolling_xi.png")
            plt.close()

            final.groupby("date")[["sigma_model"]].mean().rolling(252).mean().plot(
                legend=False
            )
            plt.ylabel(r"$\sigma$")
            plt.xlabel("Date")
            plt.savefig(d + "rolling_sigma.png")
            plt.close()

            plt.hist(
                final["xi_model"].values, bins=50, density=True, color="k", alpha=0.8
            )
            plt.xlabel(r"$\xi$")
            plt.savefig(d + "hist_xi.png")
            plt.close()

            plt.hist(
                final["sigma_model"].values, bins=50, density=True, color="k", alpha=0.8
            )
            plt.xlabel(r"$\sigma$")
            plt.savefig(d + "hist_sigma.png")
            plt.close()

            plt.scatter(
                final["sigma_model"].values,
                final["xi_model"].values,
                color="k",
                alpha=0.8,
                marker="+",
            )
            plt.xlabel(r"$\sigma$")
            plt.grid()
            plt.savefig(d + "scatter.png")
            plt.close()
        except:
            print("problem with hist")

    def produce_scatter_plots(self, final: pd.DataFrame):
        d = os.path.join(self.model.res_dir, "scatter")
        if not os.path.exists(d):
            os.makedirs(d)
        plt.scatter(final["y"], final["pred"], color="k", marker="+", alpha=0.8)
        plt.ylabel("forcasted mean of log(y)")
        plt.xlabel("log(y)")
        plt.tight_layout()
        plt.savefig(d + "mean_scatter.png")
        plt.close()

        plt.scatter(
            final["y"] ** 2, final["s_pred"] ** 2, color="k", marker="+", alpha=0.8
        )
        plt.ylabel("forcasted mean of log(y)")
        plt.xlabel("log(y)")
        plt.tight_layout()
        plt.savefig(d + "std_scatter.png")
        plt.close()

        t = final.groupby("ticker")[["y", "pred", "s_pred"]].mean()
        correlation_mean_firm = t["y"].corr(t["pred"])
        correlation_mean_firm = round(correlation_mean_firm, 3)
        plt.scatter(t["y"], t["pred"], color="k", marker="+", alpha=0.8)
        plt.ylabel("Firm average, forcasted mean $y_{i,t}$")
        plt.xlabel("Firm average, $y_{i_t}$")
        plt.tight_layout()
        plt.savefig(d + f"mean_scatter_firm_corr={correlation_mean_firm}.png")
        plt.close()

        t_s = final.groupby("ticker")[["y", "pred"]].std()
        plt.scatter(t_s["y"], t["s_pred"], color="k", marker="+", alpha=0.8)
        plt.ylabel("Firm average, forcasted STD")
        plt.xlabel("Firm average, std(log(y))")
        plt.tight_layout()
        plt.savefig(d + "std_scatter_firm.png")
        plt.close()

        t_s["log_y"] = np.log(t_s["y"])
        t_s["log_s_pred"] = np.log(t["s_pred"])
        correlation_s_pred = t_s["log_y"].corr(t_s["log_s_pred"])
        correlation_s_pred = round(correlation_s_pred, 3)
        plt.scatter(t_s["log_y"], t_s["log_s_pred"], color="k", marker="+", alpha=0.8)
        plt.ylabel("Log firm average, forcasted std")
        plt.xlabel("Log firm average, std($y_{i,t}$)")
        plt.tight_layout()
        plt.savefig(d + f"log_std_scatter_firm_corr={correlation_s_pred}.png")
        plt.close()

        final["qt"] = pd.qcut(
            final["pred"], 100, np.arange(0, 100, 1), duplicates="drop"
        ).values.tolist()
        final["qt"] = pd.qcut(final["pred"], 100, duplicates="drop").values.tolist()
        final["q_mean"] = final["qt"].apply(lambda x: (x.left + x.right) / 2)
        final.groupby("q_mean")["y"].mean().plot(color="k")
        plt.xlabel("Percentile mean forecasted")
        plt.ylabel("Mean realized log return")
        plt.grid()
        plt.tight_layout()
        plt.savefig(d + "percentile_mean.png")
        plt.close()

        final["qt"] = pd.qcut(final["pred"], 10, duplicates="drop").values.tolist()
        final["q_mean"] = final["qt"].apply(lambda x: (x.left + x.right) / 2)
        final.groupby("q_mean")["y"].mean().plot(color="k", marker="o")
        plt.xlabel("Quantile mean forecasted")
        plt.ylabel("Mean realized log return")
        plt.grid()
        plt.tight_layout()
        plt.savefig(d + "quantile_mean.png")
        plt.close()
        final["qt"] = pd.qcut(final["s_pred"], 10, duplicates="drop").values.tolist()
        final["qt"] = final["qt"].apply(lambda x: (x.left + x.right) / 2)
        quantile = final.groupby("qt")["y"].std()
        quantile = quantile.reset_index(drop=True)
        quantile.index = list(range(1, quantile.shape[0] + 1))
        quantile.plot(color="k", marker="o")
        plt.xlabel("Quantile std forecasted")
        plt.ylabel("STD realized log return")
        plt.grid()
        plt.tight_layout()
        plt.savefig(d + "quantile_std.png")
        plt.close()
