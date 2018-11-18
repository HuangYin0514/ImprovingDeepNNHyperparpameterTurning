# -*- coding: utf-8 -*-
# @Time     : 2018/11/2 21:52
# @Author   : HuangYin
# @FileName : Relu.py
# @Software : PyCharm
import numpy as np

def relu(z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    a = np.maximum(0, z)
    assert (a.shape == z.shape)
    cache = z
    return a, cache
