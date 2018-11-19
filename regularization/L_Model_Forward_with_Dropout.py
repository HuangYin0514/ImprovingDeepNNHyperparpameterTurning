# -*- coding: utf-8 -*-
# @Time     : 2018/11/18 18:43
# @Author   : HuangYin
# @FileName : L_Model_Forward_with_Dropout.py
# @Software : PyCharm

import numpy as np
from Relu import relu
from Sigmoid import sigmoid


def L_model_forward_with_dropout(X, parameters, keep_prob=0.5):
    """
       Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

       Arguments:
       X -- input dataset, of shape (2, number of examples)
       parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                       W1 -- weight matrix of shape (20, 2)
                       b1 -- bias vector of shape (20, 1)
                       W2 -- weight matrix of shape (3, 20)
                       b2 -- bias vector of shape (3, 1)
                       W3 -- weight matrix of shape (1, 3)
                       b3 -- bias vector of shape (1, 1)
       keep_prob - probability of keeping a neuron active during drop-out, scalar

       Returns:
       A3 -- last activation value, output of the forward propagation, of shape (1,1)
       cache -- tuple, information stored for computing the backward propagation
       """
    np.random.seed(1)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1,cache = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = D1 < keep_prob
    A1 = A1 * D1
    A1 = A1 / keep_prob

    Z2 = np.dot(W2, A1) + b2
    A2,cache = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 /= keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3,cache = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache
