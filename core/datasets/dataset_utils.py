import torch


def mean0_var1(x, mean, std):
    x = (x - mean) / std
    return x
