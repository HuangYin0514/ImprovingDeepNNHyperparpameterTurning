# -*- coding: utf-8 -*-
# @Time     : 2018/11/18 18:43
# @Author   : HuangYin
# @FileName : L_Model_Backward_with_Dropout.py
# @Software : PyCharm

import numpy as np


def L_model_Backward_with_dropout(X, Y, caches, keep_prob):
    """
     Implements the backward propagation of our baseline model to which we added dropout.

     Arguments:
     X -- input dataset, of shape (2, number of examples)
     Y -- "true" labels vector, of shape (output size, number of examples)
     cache -- cache output from forward_propagation_with_dropout()
     keep_prob - probability of keeping a neuron active during drop-out, scalar

     Returns:
     gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
     """
    gradients = {}
    m = Y.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = caches

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A3.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dA2 = dA2 * D2
    dA2 /= keep_prob
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1 * D1
    dA1 /= keep_prob
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients
