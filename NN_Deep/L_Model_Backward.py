# -*- coding: utf-8 -*-
# @Time     : 2018/11/3 15:14
# @Author   : HuangYin
# @FileName : L_Model_Backward.py
# @Software : PyCharm
import numpy as np
from Linear_Activation_Backward import linear_activation_backward


def L_model_Backward(AL, Y, caches):
    """
      Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

      Arguments:
      AL -- probability vector, output of the forward propagation (L_model_forward())
      Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
      caches -- list of caches containing:
                  every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                  the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

      Returns:
      grads -- A dictionary with the gradients
               grads["dA" + str(l)] = ...
               grads["dW" + str(l)] = ...
               grads["db" + str(l)] = ...
      """
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    # cost dAL
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # compute sigmoid backward
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  "sigmoid")
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        grads["dA" + str(l+1)], grads["dW" + str(l+1)], grads["db" + str(l+1)] = linear_activation_backward(grads["dA" + str(l+2)],
                                                                                                      current_cache,
                                                                                                      "relu")

    return grads
