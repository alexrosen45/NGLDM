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


NOISE = {"normal": normal, "gumbel": gumbel, "logistic": logistic, "exponential": exponential}