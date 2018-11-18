# -*- coding: utf-8 -*-
# @Time     : 2018/11/2 21:19
# @Author   : HuangYin
# @FileName : Initialize_Parameters_Deep.py
# @Software : PyCharm
import numpy as np


def initialize_parameters_deep(layer_dims):
    """
       Arguments:
       layer_dims -- python array (list) containing the dimensions of each layer in our network

       Returns:
       parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                       Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                       bl -- bias vector of shape (layer_dims[l], 1)
       """
    np.random.seed(1)
    L = len(layer_dims)
    parameters = {}

    for l in range(1, L):
        # 参数初始化不太大(在0附近),保证算法有很快的梯度下降
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert (parameters["W" + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters["b" + str(l)].shape == (layer_dims[l], 1))

    return parameters
