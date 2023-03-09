import os
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import pandas as pd
from parameters.parameters import ParamsModels, Loss, BoostType, Optimizer


class NetworkModel:
    def __init__(
        self,
        par: ParamsModels,
        par_name: str,
        save_dir: str = "save-dir",
    ):
        self.par = par
        self.model = None
        self.par.save_dir = save_dir

        self.save_dir = os.path.join(self.par.save_dir, par_name)

        self.log_dir = os.path.join(self.par.log_dir, par_name)
        self.par.res_dir = save_dir
        self.res_dir = os.path.join(self.par.res_dir, par_name)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.res_dir, exist_ok=True)

    def normalize(self, X):
        if self.par.normalize:
            X = (X - self.m_s) / self.v_s
        return X

    def train(self, X, y, data=None):
        if self.model is None:
            self.create_nnet_model()

        val = (data.X_test_val, data.y_test_val)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.save_dir + "/",
            save_weights_only=True,
            verbose=0,
            save_best_only=True,
        )

        if self.par.boost == BoostType.UNIFORM:
            # y_s = y
            X_s = []
            y_s = []
            for i in range(1):
                y_s.append(y + np.random.uniform(low=-0.02, high=0.02, size=y.shape))
                X_s.append(X + np.random.uniform(low=-0.05, high=0.05, size=X.shape))
            X_s = X_s + X
            y_s = y_s + y
            y = np.concatenate(y_s)
            X = np.concatenate(X_s)
            y[y < 0] = 0

        if self.par.boost == BoostType.EXTREME_QUANTILE:
            # print('HERE')
            qcut = 0.1

            # X=self.data.X_train; y=self.data.y_train
            ind = y[:, 0] <= np.quantile(y, qcut)
            X_l = X[ind, :]
            y_l = y[ind, :] + np.random.uniform(
                low=-0.01, high=0.01, size=(np.sum(ind), 1)
            )

            ind = y[:, 0] >= np.quantile(y, 1 - qcut)
            X_u = X[ind, :]
            y_u = y[ind, :] + np.random.uniform(
                low=-0.01, high=0.01, size=(np.sum(ind), 1)
            )

            for i in range(4):
                y = np.concatenate([y, y_l, y_u])
                X = np.concatenate([X, X_l, X_u])

        if self.par.boost == BoostType.NEGATIVE_DOUBLE:
            # print('HERE')

            # X=self.data.X_train; y=self.data.y_train
            ind = y[:, 0] <= 0.5
            X_neg = X[ind, :]
            y_neg = y[ind, :] + np.random.uniform(
                low=-0.01, high=0.01, size=(np.sum(ind), 1)
            )
            y = np.concatenate([y, y_neg])
            X = np.concatenate([X, X_neg])

        ind = np.arange(0, len(y), 1)
        np.random.shuffle(ind)

        X = X[ind, :]
        y = y[ind]

        print("start training for", self.par.E, "epochs", flush=True)
        # old version with tensorboard
        if data is None:
            self.history_training = self.model.fit(
                x=X,
                y=y,
                batch_size=self.par.batch_size,
                epochs=self.par.E,
                validation_split=self.par.validation_split,
                callbacks=[cp_callback],
                verbose=1,
            )  # Pass callback to training
        else:
            self.history_training = self.model.fit(
                x=X,
                y=y,
                validation_data=val,
                batch_size=self.par.batch_size,
                epochs=self.par.E,
                callbacks=[cp_callback],
                verbose=1,
            )  # Pass callback to training

        self.history_training = pd.DataFrame(self.history_training.history)

    def score(self, X, y):
        X = self.normalize(X)
        val = self.model.evaluate(X, y, verbose=0)
        return val

    def x_to_law_input(self, X):
        X = self.normalize(X)
        pred = self.model.predict(X)
        return pred

    def x_to_human_input(self, X):
        X = self.normalize(X)
        input_law = self.model.predict(X)

        if self.par.loss in [Loss.EX_GDP_SIGMA]:
            EPSILON = 0.000001
            xi = tf.nn.sigmoid(input_law[:, 0]) * 1.0 + EPSILON
            sigma = tf.nn.sigmoid(input_law[:, 1]) * 0.5 + EPSILON

            RES = pd.DataFrame(data={"sigma": sigma, "xi": xi})

        if self.par.loss in [Loss.PARETO_ONLY_QUANTILE, Loss.PARETO_ONLY_SIGMA]:
            EPSILON = 0.000001
            xi = tf.nn.sigmoid(input_law[:, 0]) * 1.5 + EPSILON
            sigma = tf.nn.sigmoid(input_law[:, 1]) * 1.5 + EPSILON

            RES = pd.DataFrame(data={"sigma": sigma, "xi": xi})

        if self.par.loss == Loss.LOG_LIKE_PARETO:
            EPSILON = 0.000001
            # mixture pareto
            sig_n = tf.nn.sigmoid(input_law[:, 0]) * 0.5 + EPSILON
            mu_n = tf.nn.tanh(input_law[:, 1]) * 0.1
            u_d = -(2 + tf.nn.sigmoid(input_law[:, 0])) * sig_n + mu_n
            u_u = (2 + tf.nn.sigmoid(input_law[:, 1])) * sig_n + mu_n

            xi_u = tf.sigmoid(input_law[:, 4]) * (0.5 - EPSILON * 2) + EPSILON
            xi_d = tf.sigmoid(input_law[:, 5]) * (0.5 - EPSILON * 2) + EPSILON

            dist = tfp.distributions.Normal(loc=mu_n, scale=sig_n)
            # sig_p_u = tf.clip_by_value(1.0 / dist.prob(u_u), 0.001, 10.0)
            # sig_p_d = tf.clip_by_value(1.0 / dist.prob(u_d), 0.001, 10.0)
            sig_p_u = tf.nn.sigmoid(input_law[:, 6]) * 1.0 + EPSILON
            sig_p_d = tf.nn.sigmoid(input_law[:, 7]) * 1.0 + EPSILON

            RES = pd.DataFrame(
                data={
                    "u_d": u_d,
                    "u_u": u_u,
                    "mu_n": mu_n,
                    "sig_n": sig_n,
                    "sig_p_u": sig_p_u,
                    "sig_p_d": sig_p_d,
                    "xi_u": xi_u,
                    "xi_d": xi_d,
                }
            )

        if self.par.loss == Loss.PARETO_FIX_SIG_DIST:
            EPSILON = 0.000001
            # mixture pareto
            sig_n = tf.abs(input_law[:, 0]) + EPSILON
            mu_n = tf.nn.tanh(input_law[:, 1]) * 0.1
            u_d = -2 * sig_n + mu_n
            u_u = 2 * sig_n + mu_n

            xi_u = tf.sigmoid(input_law[:, 2]) * (0.5 - EPSILON * 2) + EPSILON
            xi_d = tf.sigmoid(input_law[:, 3]) * (0.5 - EPSILON * 2) + EPSILON

            sig_p_u = tf.abs(input_law[:, 4]) + EPSILON
            sig_p_d = tf.abs(input_law[:, 5]) + EPSILON

            RES = pd.DataFrame(
                data={
                    "u_d": u_d,
                    "u_u": u_u,
                    "mu_n": mu_n,
                    "sig_n": sig_n,
                    "sig_p_u": sig_p_u,
                    "sig_p_d": sig_p_d,
                    "xi_u": xi_u,
                    "xi_d": xi_d,
                }
            )

        if self.par.loss == Loss.PARETO_SIG_DIST:
            EPSILON = 0.000001
            # mixture pareto
            sig_n = tf.abs(input_law[:, 0]) + EPSILON
            mu_n = tf.nn.tanh(input_law[:, 1]) * 0.1
            u_d = -(2 * tf.nn.sigmoid(input_law[:, 2])) * sig_n + mu_n
            u_u = (2 * tf.nn.sigmoid(input_law[:, 3])) * sig_n + mu_n

            xi_u = tf.sigmoid(input_law[:, 4]) * (0.5 - EPSILON * 2) + EPSILON
            xi_d = tf.sigmoid(input_law[:, 5]) * (0.5 - EPSILON * 2) + EPSILON

            sig_p_u = tf.abs(input_law[:, 6]) + EPSILON
            sig_p_d = tf.abs(input_law[:, 7]) + EPSILON

            RES = pd.DataFrame(
                data={
                    "u_d": u_d,
                    "u_u": u_u,
                    "mu_n": mu_n,
                    "sig_n": sig_n,
                    "sig_p_u": sig_p_u,
                    "sig_p_d": sig_p_d,
                    "xi_u": xi_u,
                    "xi_d": xi_d,
                }
            )

        if self.par.loss == Loss.PARETO_FIX_U:
            sig_start = 0.1
            EPSILON = 0.000001
            # mixture pareto
            sig_n = tf.abs(input_law[:, 0]) + EPSILON
            mu_n = tf.nn.tanh(input_law[:, 1]) * 0.1
            u_d = -sig_start
            u_u = sig_start

            xi_u = tf.sigmoid(input_law[:, 2]) * (0.5 - EPSILON * 2) + EPSILON
            xi_d = tf.sigmoid(input_law[:, 3]) * (0.5 - EPSILON * 2) + EPSILON

            sig_p_u = tf.abs(input_law[:, 4]) + EPSILON
            sig_p_d = tf.abs(input_law[:, 5]) + EPSILON

            RES = pd.DataFrame(
                data={
                    "u_d": u_d,
                    "u_u": u_u,
                    "mu_n": mu_n,
                    "sig_n": sig_n,
                    "sig_p_u": sig_p_u,
                    "sig_p_d": sig_p_d,
                    "xi_u": xi_u,
                    "xi_d": xi_d,
                }
            )

        if self.par.loss in [Loss.LOG_LIKE_RESTRICTED, Loss.LOG_LIKE]:
            # some kind of mixture of gaussian
            nb_normal = int(input_law.shape[1] / 3)
            PI = tf.nn.softmax(input_law[:, :nb_normal]).numpy()

            if self.par.loss == Loss.LOG_LIKE:
                MU = [input_law[:, nb_normal]]
                SIG = [tf.abs(input_law[:, nb_normal + 1]).numpy() + 0.001]
                for i in range(1, nb_normal):
                    MU.append(input_law[:, nb_normal + 2 * i])
                    SIG.append(
                        tf.abs(input_law[:, nb_normal + (2 * i) + 1]).numpy() + 0.001
                    )

            if self.par.loss == Loss.LOG_LIKE_RESTRICTED:
                MU = [input_law[:, nb_normal]]
                for i in range(1, nb_normal):
                    MU.append(MU[i - 1] + tf.abs(input_law[:, nb_normal + 2 * i]))
                SIG_T = [0.001 + tf.abs(input_law[:, nb_normal + 1])]
                for i in range(1, nb_normal):
                    SIG_T.append(
                        SIG_T[i - 1] + tf.abs(input_law[:, nb_normal + (2 * i) + 1])
                    )
                SIG = []
                for i in range(0, nb_normal):
                    SIG.append(SIG_T[nb_normal - 1 - i])
            RES = np.stack(MU, 1), np.stack(SIG, 1), np.stack(PI, 1).T
        return RES

    def law_input_to_probability(self, law_input, y):
        if len(y.shape) < 2:
            y_ = y.reshape(-1, 1)
        else:
            y_ = y
        r = NetworkModel.get_prop(law_input, y_)
        return r

    def x_to_probability(self, X, y):
        law_input = self.x_to_law_input(X)
        r = self.law_input_to_probability(law_input, y)
        return r

    def create_nnet_model(self):
        L = []
        for i, l in enumerate(self.par.layers):
            L.append(
                tf.keras.layers.Dense(
                    l, activation=self.par.activation, dtype=tf.float32
                )
            )
            if self.par.dropout != 0.0:
                L.append(tf.keras.layers.Dropout(self.par.dropout))

        # add the final layer
        if self.par.loss in [
            Loss.PARETO_ONLY_SIGMA,
            Loss.PARETO_ONLY_QUANTILE,
            Loss.EX_GDP_SIGMA,
        ]:
            f_dim = 2
            L.append(
                tf.keras.layers.Dense(f_dim, activation="linear", dtype=tf.float32)
            )
        # add the final layer
        if self.par.loss == Loss.LOG_LIKE_PARETO:
            f_dim = 8
            L.append(
                tf.keras.layers.Dense(f_dim, activation="linear", dtype=tf.float32)
            )
        if self.par.loss == Loss.PARETO_FIX_SIG_DIST:
            f_dim = 6
            L.append(
                tf.keras.layers.Dense(f_dim, activation="linear", dtype=tf.float32)
            )
        if self.par.loss == Loss.PARETO_SIG_DIST:
            f_dim = 8
            L.append(
                tf.keras.layers.Dense(f_dim, activation="linear", dtype=tf.float32)
            )
        if self.par.loss == Loss.PARETO_FIX_U:
            f_dim = 6
            L.append(
                tf.keras.layers.Dense(f_dim, activation="linear", dtype=tf.float32)
            )
        if self.par.loss in [Loss.LOG_LIKE, Loss.LOG_LIKE_RESTRICTED]:
            # else we are in a mixture of gaussian
            f_dim = self.par.nb_normal * 3
            L.append(
                tf.keras.layers.Dense(f_dim, activation="linear", dtype=tf.float32)
            )
        if self.par.loss in [Loss.MAE, Loss.MSE, Loss.MSE_VOLA, Loss.R2]:
            f_dim = 1
            L.append(
                tf.keras.layers.Dense(f_dim, activation="linear", dtype=tf.float32)
            )
        if self.par.loss in [Loss.SIGN, Loss.PRB_QUANTILE, Loss.PRB_SIGMA_DIST]:
            # else we are in a mixture of gaussian
            f_dim = 1
            L.append(
                tf.keras.layers.Dense(f_dim, activation="sigmoid", dtype=tf.float32)
            )
        self.model = tf.keras.Sequential(L)

        # optimizer = tf.keras.optimizers.RMSprop(0.05)
        if self.par.opti == Optimizer.SGD:
            optimizer = tf.keras.optimizers.SGD(self.par.learning_rate)
        if self.par.opti == Optimizer.RMS_PROP:
            optimizer = tf.keras.optimizers.RMSprop(self.par.learning_rate)
        if self.par.opti == Optimizer.ADAM:
            optimizer = tf.keras.optimizers.Adam(self.par.learning_rate)

        def r_square(y_true, y_pred):
            SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
            SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
            return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

        if self.par.loss == Loss.LOG_LIKE:
            self.model.compile(loss=NetworkModel.log_like, optimizer=optimizer)
        if self.par.loss == Loss.LOG_LIKE_RESTRICTED:
            self.model.compile(
                loss=NetworkModel.log_like_restricted, optimizer=optimizer
            )
        if self.par.loss == Loss.LOG_LIKE_PARETO:
            self.model.compile(loss=NetworkModel.log_like_pareto, optimizer=optimizer)
        if self.par.loss == Loss.PARETO_FIX_U:
            self.model.compile(
                loss=NetworkModel.log_like_pareto_fix_u, optimizer=optimizer
            )
        if self.par.loss == Loss.PARETO_SIG_DIST:
            self.model.compile(
                loss=NetworkModel.log_like_pareto_sig_dist, optimizer=optimizer
            )
        if self.par.loss == Loss.PARETO_FIX_SIG_DIST:
            self.model.compile(
                loss=NetworkModel.log_like_pareto_fix_sig_dist, optimizer=optimizer
            )
        if self.par.loss == Loss.MAE:
            self.model.compile(
                loss=tf.losses.mae, optimizer=optimizer, metrics=[r_square]
            )
        if self.par.loss in [Loss.MSE, Loss.MSE_VOLA]:
            self.model.compile(
                loss=tf.losses.mse, optimizer=optimizer, metrics=[r_square]
            )
        if self.par.loss in [Loss.SIGN, Loss.PRB_QUANTILE, Loss.PRB_SIGMA_DIST]:
            self.model.compile(loss=tf.losses.binary_crossentropy, optimizer=optimizer)
        if self.par.loss == Loss.R2:
            self.model.compile(loss=r_square, optimizer=optimizer)
        if self.par.loss in [Loss.PARETO_ONLY_QUANTILE, Loss.PARETO_ONLY_SIGMA]:
            self.model.compile(
                loss=NetworkModel.log_like_pareto_only, optimizer=optimizer
            )
        if self.par.loss in [Loss.EX_GDP_SIGMA]:
            self.model.compile(loss=NetworkModel.log_exgdp, optimizer=optimizer)

    def multiprint(self, text, file_=None):
        file_ = self.res_dir + "/out.txt"
        print(text, flush=True)
        print(text, file=open(file_, "a"))

    ##################
    # cost functions, exp gdpo
    ##################

    @staticmethod
    @tf.function
    def likelihood_exgdp(x):
        EPSILON = 0.000001
        xi = tf.nn.sigmoid(x[0]) * 1.5 + EPSILON
        # xi = tf.nn.sigmoid(x[0]) * (0.5 - EPSILON * 2) + EPSILON
        sigma = tf.nn.sigmoid(x[1]) * 1.5 + EPSILON
        y = x[2]

        e = np.exp(y)
        return (e / sigma) * (1 + xi * e / sigma) ** ((-1 / xi) - 1)

    @staticmethod
    def get_prop_exgdp(input_law, y_true):
        m = tf.concat((input_law, y_true), axis=1)
        p = tf.vectorized_map(fn=NetworkModel.likelihood_pareto_only, elems=m)
        return p

    @staticmethod
    def log_exgdp(y_true, y_pred):
        EPSILON = 0.000001
        e = tf.exp(y_true[:, 0])
        xi = tf.nn.sigmoid(y_pred[:, 0]) * 1.0 + EPSILON
        sigma = tf.nn.sigmoid(y_pred[:, 1]) * 0.5 + EPSILON

        # r = (e / sigma) * (1 + xi * e / sigma) ** ((-1 / xi) - 1)
        a = e / sigma
        b = 1 + xi * e / sigma
        c = (-1 / xi) - 1
        r = a * tf.pow(b, c)
        r = -tf.math.log(r + 1e-10)
        return tf.reduce_mean(r)

    ##################
    # cost functions, pareto only
    ##################

    @staticmethod
    @tf.function
    def likelihood_pareto_only(x):
        EPSILON = 0.000001
        xi = tf.nn.sigmoid(x[0]) * 1.5 + EPSILON
        # xi = tf.nn.sigmoid(x[0]) * (0.5 - EPSILON * 2) + EPSILON
        sigma = tf.nn.sigmoid(x[1]) * 1.5 + EPSILON
        y = x[2]

        d = tfp.distributions.GeneralizedPareto(0.0, sigma, xi)
        r = d.prob(y)
        return r

    @staticmethod
    def get_prop_pareto_only(input_law, y_true):
        m = tf.concat((input_law, y_true), axis=1)
        p = tf.vectorized_map(fn=NetworkModel.likelihood_pareto_only, elems=m)
        return p

    @staticmethod
    def log_like_pareto_only(y_true, y_pred):
        r = NetworkModel.get_prop_pareto_only(y_pred, y_true)
        r = -tf.math.log(r + 1e-10)
        return tf.reduce_mean(r)

    ##################
    # cost function with pareto
    ##################

    @staticmethod
    def pareto_pdf(x, xi, mu, sigma):
        # -y, xi_d, u_d, sig_p_d
        # x=y; xi = xi_u; mu = u_u; sigma = sig_p_u
        z = (x - mu) / sigma
        a = (1 / sigma) * (1 + xi * z) ** (-1 / (xi + 1))
        return a

    @staticmethod
    @tf.function
    def likelihood_mixture_pareto(x):
        EPSILON = 0.000001

        sig_n = tf.abs(x[0]) + EPSILON

        mu_n = tf.nn.tanh(x[1]) * 0.1
        u_d = -(2 + tf.nn.sigmoid(x[2])) * sig_n + mu_n
        u_u = (2 + tf.nn.sigmoid(x[3])) * sig_n + mu_n

        xi_u = tf.nn.sigmoid(x[4]) * (0.5 - EPSILON * 2) + EPSILON
        xi_d = tf.nn.sigmoid(x[5]) * (0.5 - EPSILON * 2) + EPSILON

        sig_p_u = tf.nn.sigmoid(x[6]) * 1.0 + EPSILON
        sig_p_d = tf.nn.sigmoid(x[7]) * 1.0 + EPSILON
        y = x[8]

        dist = tfp.distributions.Normal(loc=mu_n, scale=sig_n)
        # sig_p_u = tf.clip_by_value(1.0 / dist.prob(u_u), 0.001, 10.0)
        # sig_p_d = tf.clip_by_value(1.0 / dist.prob(u_d), 0.001, 10.0)

        f = 1.0 / (dist.cdf(u_u) - dist.cdf(u_d) + 2.0)

        d_p_u = tfp.distributions.GeneralizedPareto(
            loc=u_u, scale=sig_p_u, concentration=xi_u
        )
        d_p_d = tfp.distributions.GeneralizedPareto(
            loc=u_d, scale=sig_p_d, concentration=xi_d
        )

        if (y > u_d) & (y < u_u):
            r = dist.prob(y) * f
        else:
            if y <= u_d:
                r = d_p_d.prob(2.0 * u_d - y) * f
            else:
                r = d_p_u.prob(y) * f

        return r

    @staticmethod
    @tf.function
    def likelihood_mixture_pareto_fix_sig_dist(x):
        EPSILON = 0.000001

        sig_n = tf.abs(x[0]) + EPSILON

        mu_n = tf.nn.tanh(x[1]) * 0.1
        u_d = -2 * sig_n + mu_n
        u_u = 2 * sig_n + mu_n

        xi_u = tf.nn.sigmoid(x[2]) * (0.5 - EPSILON * 2) + EPSILON
        xi_d = tf.nn.sigmoid(x[3]) * (0.5 - EPSILON * 2) + EPSILON

        sig_p_u = tf.abs(x[4]) + EPSILON
        sig_p_d = tf.abs(x[5]) + EPSILON
        y = x[6]

        dist = tfp.distributions.Normal(loc=mu_n, scale=sig_n)

        f = 1.0 / (dist.cdf(u_u) - dist.cdf(u_d) + 2.0)

        d_p_u = tfp.distributions.GeneralizedPareto(
            loc=u_u, scale=sig_p_u, concentration=xi_u
        )
        d_p_d = tfp.distributions.GeneralizedPareto(
            loc=u_d, scale=sig_p_d, concentration=xi_d
        )

        if (y > u_d) & (y < u_u):
            r = dist.prob(y) * f
        else:
            if y <= u_d:
                r = d_p_d.prob(2.0 * u_d - y) * f
            else:
                r = d_p_u.prob(y) * f
        return r

    @staticmethod
    @tf.function
    def likelihood_mixture_pareto_fix_sig_dist(x):
        EPSILON = 0.000001

        sig_n = tf.abs(x[0]) + EPSILON

        mu_n = tf.nn.tanh(x[1]) * 0.1
        u_d = -2 * sig_n + mu_n
        u_u = 2 * sig_n + mu_n

        xi_u = tf.nn.sigmoid(x[2]) * (0.5 - EPSILON * 2) + EPSILON
        xi_d = tf.nn.sigmoid(x[3]) * (0.5 - EPSILON * 2) + EPSILON

        sig_p_u = tf.abs(x[4]) + EPSILON
        sig_p_d = tf.abs(x[5]) + EPSILON
        y = x[6]

        dist = tfp.distributions.Normal(loc=mu_n, scale=sig_n)

        f = 1.0 / (dist.cdf(u_u) - dist.cdf(u_d) + 2.0)

        d_p_u = tfp.distributions.GeneralizedPareto(
            loc=u_u, scale=sig_p_u, concentration=xi_u
        )
        d_p_d = tfp.distributions.GeneralizedPareto(
            loc=u_d, scale=sig_p_d, concentration=xi_d
        )

        if (y > u_d) & (y < u_u):
            r = dist.prob(y) * f
        else:
            if y <= u_d:
                r = d_p_d.prob(2.0 * u_d - y) * f
            else:
                r = d_p_u.prob(y) * f
        return r

    @staticmethod
    @tf.function
    def likelihood_mixture_pareto_sig_dist(x):
        EPSILON = 0.000001

        sig_n = tf.abs(x[0]) + EPSILON

        mu_n = tf.nn.tanh(x[1]) * 0.1
        u_d = -(2 * tf.nn.sigmoid(x[2])) * sig_n + mu_n
        u_u = (2 * tf.nn.sigmoid(x[3])) * sig_n + mu_n

        xi_u = tf.nn.sigmoid(x[4]) * (0.5 - EPSILON * 2) + EPSILON
        xi_d = tf.nn.sigmoid(x[5]) * (0.5 - EPSILON * 2) + EPSILON

        sig_p_u = tf.abs(x[6]) + EPSILON
        sig_p_d = tf.abs(x[7]) + EPSILON
        y = x[8]

        dist = tfp.distributions.Normal(loc=mu_n, scale=sig_n)

        f = 1.0 / (dist.cdf(u_u) - dist.cdf(u_d) + 2.0)

        d_p_u = tfp.distributions.GeneralizedPareto(
            loc=u_u, scale=sig_p_u, concentration=xi_u
        )
        d_p_d = tfp.distributions.GeneralizedPareto(
            loc=u_d, scale=sig_p_d, concentration=xi_d
        )

        if (y > u_d) & (y < u_u):
            r = dist.prob(y) * f
        else:
            if y <= u_d:
                r = d_p_d.prob(2.0 * u_d - y) * f
            else:
                r = d_p_u.prob(y) * f
        return r

    @staticmethod
    @tf.function
    def likelihood_mixture_pareto_fix_u(x):
        sig_start = 0.1
        EPSILON = 0.000001

        sig_n = tf.abs(x[0]) + EPSILON

        mu_n = tf.nn.tanh(x[1]) * (sig_start - EPSILON)
        u_d = -sig_start
        u_u = sig_start

        xi_u = tf.nn.sigmoid(x[2]) * (0.5 - EPSILON * 2) + EPSILON
        xi_d = tf.nn.sigmoid(x[3]) * (0.5 - EPSILON * 2) + EPSILON

        sig_p_u = tf.abs(x[4]) + EPSILON
        sig_p_d = tf.abs(x[5]) + EPSILON
        y = x[6]

        dist = tfp.distributions.Normal(loc=mu_n, scale=sig_n)

        f = 1.0 / (dist.cdf(u_u) - dist.cdf(u_d) + 2.0)

        d_p_u = tfp.distributions.GeneralizedPareto(
            loc=u_u, scale=sig_p_u, concentration=xi_u
        )
        d_p_d = tfp.distributions.GeneralizedPareto(
            loc=u_d, scale=sig_p_d, concentration=xi_d
        )

        if (y > u_d) & (y < u_u):
            r = dist.prob(y) * f
        else:
            if y <= u_d:
                r = d_p_d.prob(2.0 * u_d - y) * f
            else:
                r = d_p_u.prob(y) * f
        return r

    @staticmethod
    def get_prop_pareto(input_law, y_true):
        m = tf.concat((input_law, y_true), axis=1)
        p = tf.vectorized_map(fn=NetworkModel.likelihood_mixture_pareto, elems=m)
        return p

    @staticmethod
    def get_prop_pareto_fix_sig_dist(input_law, y_true):
        m = tf.concat((input_law, y_true), axis=1)
        p = tf.vectorized_map(
            fn=NetworkModel.likelihood_mixture_pareto_fix_sig_dist, elems=m
        )
        return p

    @staticmethod
    def get_prop_pareto_sig_dist(input_law, y_true):
        m = tf.concat((input_law, y_true), axis=1)
        p = tf.vectorized_map(
            fn=NetworkModel.likelihood_mixture_pareto_sig_dist, elems=m
        )
        return p

    @staticmethod
    def get_prop_pareto_fix_u(input_law, y_true):
        m = tf.concat((input_law, y_true), axis=1)
        p = tf.vectorized_map(fn=NetworkModel.likelihood_mixture_pareto_fix_u, elems=m)
        return p

    @staticmethod
    def log_like_pareto(y_true, y_pred):
        r = NetworkModel.get_prop_pareto(y_pred, y_true)
        r = -tf.math.log(r + 1e-10)
        return tf.reduce_mean(r)

    @staticmethod
    def log_like_pareto_fix_sig_dist(y_true, y_pred):
        r = NetworkModel.get_prop_pareto_fix_sig_dist(y_pred, y_true)
        r = -tf.math.log(r + 1e-10)
        return tf.reduce_mean(r)

    @staticmethod
    def log_like_pareto_sig_dist(y_true, y_pred):
        r = NetworkModel.get_prop_pareto_sig_dist(y_pred, y_true)
        r = -tf.math.log(r + 1e-10)
        return tf.reduce_mean(r)

    @staticmethod
    def log_like_pareto_fix_u(y_true, y_pred):
        r = NetworkModel.get_prop_pareto_fix_u(y_pred, y_true)
        r = -tf.math.log(r + 1e-10)
        return tf.reduce_mean(r)

    ##################
    # cost function with mixture of gaussian
    ##################

    @staticmethod
    def get_prop(input_law, y_true):
        nb_normal = int(input_law.shape[1] / 3)
        w = tf.nn.softmax(input_law[:, :nb_normal])

        p = 0
        for i in range(0, nb_normal):
            p += (
                NetworkModel.pr_norm(
                    x=y_true[:, 0],
                    mu=input_law[:, nb_normal + 2 * i],
                    sig=0.001 + tf.abs(input_law[:, nb_normal + (2 * i) + 1]),
                )
                * w[:, i]
            )
        return p

    @staticmethod
    def get_prop_restricted(input_law, y_true):
        nb_normal = int(input_law.shape[1] / 3)
        w = tf.nn.softmax(input_law[:, :nb_normal])

        MU = [input_law[:, nb_normal]]
        for i in range(1, nb_normal):
            MU.append(MU[i - 1] + tf.abs(input_law[:, nb_normal + 2 * i]))
        SIG_T = [0.001 + tf.abs(input_law[:, nb_normal + 1])]
        for i in range(1, nb_normal):
            SIG_T.append(SIG_T[i - 1] + tf.abs(input_law[:, nb_normal + (2 * i) + 1]))
        SIG = []
        for i in range(0, nb_normal):
            SIG.append(SIG_T[nb_normal - 1 - i])

        p = 0
        for i in range(0, nb_normal):
            p += NetworkModel.pr_norm(x=y_true[:, 0], mu=MU[i], sig=SIG[i]) * w[:, i]
        return p

    @staticmethod
    def log_like(y_true, y_pred):
        r = NetworkModel.get_prop(y_pred, y_true)
        r = -tf.math.log(r + 1e-10)
        return tf.reduce_mean(r)

    @staticmethod
    def log_like_restricted(y_true, y_pred):
        r = NetworkModel.get_prop_restricted(y_pred, y_true)
        r = -tf.math.log(r + 1e-10)
        return tf.reduce_mean(r)

    @staticmethod
    def pr_norm(x, mu, sig):
        return tf.exp(-0.5 * tf.square((x - mu) / sig)) / (sig * tf.sqrt(2 * np.pi))

    ##################
    # save load
    ##################

    def load(self, n):
        # self.par.name = n
        temp_dir = self.par.save_dir + "" + n
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # par = Params()
        # par.load(load_dir=temp_dir)

        if self.model is None:
            self.create_nnet_model()
        self.model.load_weights(self.save_dir + "/")
        print("model loaded")

    def save(self, save_dir: str, save_name: str):
        """saves the model"""
        save_path = os.path.join(save_dir, save_name)
        tf.saved_model.save(self.model, save_path)
