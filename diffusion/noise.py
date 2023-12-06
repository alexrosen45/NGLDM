import math
import numpy as np
import torch


def normal(shape, device):
    dist = torch.distributions.Normal(loc=0, scale=1)
    return dist.sample(shape).to(device)


def gumbel(shape, device, euler_mash= 0.5772):
    loc, scale = 1, 2
    dist = torch.distributions.Gumbel(loc, scale)
    sample = dist.sample(shape).to(device)
    mean = loc + euler_mash * scale
    std_dev = (math.pi / math.sqrt(6)) * scale
    return (sample - mean) / std_dev


def exponential(shape, device):
    dist = torch.distributions.Exponential(rate=1)
    return dist.sample(shape).to(device)


def laplace(shape, device):
    var = math.sqrt(1 / 2)
    dist = torch.distributions.Laplace(loc=0, scale=var)
    return dist.sample(shape).to(device)


def uniform(shape, device):
    low, high = -math.sqrt(3), math.sqrt(3)
    dist = torch.distributions.Uniform(low=low, high=high)
    return dist.sample(shape).to(device)
