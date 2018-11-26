# -*- coding: utf-8 -*-
# @Time     : 2018/11/22 8:19
# @Author   : HuangYin
# @FileName : Update_Parameters_wtih_GD.py
# @Software : PyCharm

import numpy as np


def update_Parameters_wtih_GD(parameters, grads, learning_rate):
    """
       Update parameters using one step of gradient descent

       Arguments:
       parameters -- python dictionary containing your parameters to be updated:
                       parameters['W' + str(l)] = Wl
                       parameters['b' + str(l)] = bl
       grads -- python dictionary containing your gradients to update each parameters:
                       grads['dW' + str(l)] = dWl
                       grads['db' + str(l)] = dbl
       learning_rate -- the learning rate, scalar.

       Returns:
       parameters -- python dictionary containing your updated parameters
       """
    L = len(parameters) // 2
    for l in range(0, L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters
