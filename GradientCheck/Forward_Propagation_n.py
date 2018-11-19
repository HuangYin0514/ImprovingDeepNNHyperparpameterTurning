# -*- coding: utf-8 -*-
# @Time     : 2018/11/19 14:46
# @Author   : HuangYin
# @FileName : Forward_Propagation_n.py
# @Software : PyCharm

import numpy as np
from Relu import relu
from Sigmoid import sigmoid


def forward_propagation_n(X, Y, parameters):
    """
       Implements the forward propagation (and computes the cost) presented in Figure 3.

       Arguments:
       X -- training set for m examples
       Y -- labels for m examples
       parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                       W1 -- weight matrix of shape (5, 4)
                       b1 -- bias vector of shape (5, 1)
                       W2 -- weight matrix of shape (3, 5)
                       b2 -- bias vector of shape (3, 1)
                       W3 -- weight matrix of shape (1, 3)
                       b3 -- bias vector of shape (1, 1)

       Returns:
       cost -- the cost function (logistic cost for one example)
       """
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b1
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b1
    A3 = sigmoid(Z3)

    logprob = np.multiply(Y, np.log(A3)) + np.multiply(1 - Y, np.log(1 - A3))
    cost = (-1. / m) * np.sum(logprob)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache
