# -*- coding: utf-8 -*-
# @Time     : 2018/11/18 18:43
# @Author   : HuangYin
# @FileName : L_Model_Backward_with_Regularization.py
# @Software : PyCharm
import numpy as np
from Linear_Activation_Backward_with_Regularization import linear_activation_backward_with_Regularization


def L_model_Backward_with_regularization(AL, Y, caches, lambd):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    # cost dAL
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # compute sigmoid backward
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward_with_Regularization(
        dAL, current_cache, "sigmoid",lambd)
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        grads["dA" + str(l + 1)], grads["dW" + str(l + 1)], grads[
            "db" + str(l + 1)] = linear_activation_backward_with_Regularization(
            grads["dA" + str(l + 2)], current_cache, "relu",lambd)

    return grads
