# -*- coding: utf-8 -*-
# @Time     : 2018/11/3 12:18
# @Author   : HuangYin
# @FileName : Relu_Backward.py
# @Software : PyCharm
import numpy as np


def relu_backward(dA, cache):
    """
        Implement the backward propagation for a single RELU unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    assert (dZ.shape == dA.shape)
    return dZ
