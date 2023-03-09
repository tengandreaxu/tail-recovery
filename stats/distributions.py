import numpy as np
import tensorflow as tf
from scipy import optimize


def exponential_gpd_distribution(y, xi: float, sigma: float):
    """Exponential Generalized Pareto Distribution"""
    exp_y = np.exp(y)
    return (exp_y / sigma) * (1 + xi * exp_y / sigma) ** ((-1 / xi) - 1)


def exponential_gpd_mle(y: np.ndarray):
    """Exponential Generalized Pareto Distribution Maximum Likelihood Estimation"""

    def ex_gpd(x):
        sigma = x[0]
        xi = x[1]
        r = exponential_gpd_distribution(y, xi, sigma)
        return -np.mean(np.log(r))

    r = optimize.minimize(ex_gpd, np.array([0.1, 0.1]), method="Nelder-Mead")
    ex_gpd(r.x)

    sigma = r.x[0]
    xi = r.x[1]
    return sigma, xi


def exponential_gpd_expected_value_negative(xi, sigma):

    return (
        np.log(-sigma / xi)
        + tf.math.digamma(1.0).numpy()
        - tf.math.digamma((-1.0 / xi) + 1).numpy()
    )


def exponential_gpd_expected_value_positive(xi, sigma):

    return (
        np.log(sigma / xi)
        + tf.math.digamma(1.0).numpy()
        - tf.math.digamma(1.0 / xi).numpy()
    )
