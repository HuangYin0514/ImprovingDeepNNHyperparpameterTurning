# -*- coding: utf-8 -*-
# @Time     : 2018/11/3 10:46
# @Author   : HuangYin
# @FileName : L_Model_Forward.py
# @Software : PyCharm

import numpy as np
from Linear_Activation_Forward import linear_activation_forward


def L_model_forward(X, parameters):
    """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

    # //的意思是 做除法之后结果是整数
    L = len(parameters) // 2
    caches = []
    A = X
    # activation of "relu"
    # hidden layer
    # implement [LINEAR --> RELU]*(L-1) times
    for l in range(1, L):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A_pre = A
        A, cache = linear_activation_forward(A_pre, W, b, "relu")
        caches.append(cache)

    # activation of "sigmoid"
    # output layer
    # implement [LINEAR -->SIGMOID]
    A_pre = A
    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    AL, cache = linear_activation_forward(A_pre, W, b, "sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches
