# -*- coding: utf-8 -*-
# @Time     : 2018/11/3 14:49
# @Author   : HuangYin
# @FileName : Linear_Activation_Backward_with_Regularization.py
# @Software : PyCharm
from Relu_Backward import relu_backward
from Sigmoid_Backward import sigmoid_backward
from Linear_Backward_with_Regularization import linear_backward_with_Regularization


def linear_activation_backward_with_Regularization(dA, cache, activation,lambd):
    """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
    linear_cacahe, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward_with_Regularization(dZ, linear_cacahe,lambd)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward_with_Regularization(dZ, linear_cacahe,lambd)

    assert (dA_prev.shape[1] == dA.shape[1])

    return dA_prev, dW, db
