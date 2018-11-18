# -*- coding: utf-8 -*-
# @Time     : 2018/11/2 21:53
# @Author   : HuangYin
# @FileName : Linear_Activation_Forward.py
# @Software : PyCharm
from Linear_Forward import linear_forward
from Sigmoid import sigmoid
from Relu import relu


def linear_activation_forward(A_prev, W, b, activation):
    """
       Implement the forward propagation for the LINEAR->ACTIVATION layer

       Arguments:
       A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
       W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
       b -- bias vector, numpy array of shape (size of the current layer, 1)
       activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

       Returns:
       A -- the output of the activation function, also called the post-activation value
       cache -- a python dictionary containing "linear_cache" and "activation_cache";
                stored for computing the backward pass efficiently
       """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache
