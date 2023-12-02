import math
import numpy as np


def normal(shape, var):
    return np.random.normal(scale=math.sqrt(var), size=shape)


def logistic(shape, var):
    s = math.sqrt((3 * var) / (math.pi ** 2))
    return np.random.logistic(loc=0, scale=s, size=shape)


def gumbel(shape, var):
    s = math.sqrt((6 * var) / (math.pi ** 2))
    return np.random.gumbel(loc=0, scale=s, size=shape) - s * 0.57721


def exponential(shape, var):
    s = math.sqrt(var)
    return np.random.exponential(scale=s, size=shape) - s


def binomial(shape, var):
    p = 0.5
    n = 4 * var
    return np.random.binomial(n=n, p=p, size=shape) - n*p


def gamma(shape, var):
    k = 2
    theta = math.sqrt(var / k)
    return np.random.gamma(shape=k, scale=theta, size=shape) - k * theta


def laplace(shape, var):
    b = math.sqrt(var / 2)
    return np.random.laplace(loc=0, scale=b, size=shape)


def poisson(shape, var):
    return np.random.poisson(lam=var, size=shape) - var


def uniform(shape, var):
    end = math.sqrt(3 * var)
    return np.random.uniform(low=-end, high=end, size=shape)


NOISE = {"normal": normal, "gumbel": gumbel, "logistic": logistic, "exponential": exponential,
         "binomial": binomial, "gamma": gamma, "laplace": laplace, "poisson": poisson, "uniform": uniform}
