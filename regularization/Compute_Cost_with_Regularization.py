# -*- coding: utf-8 -*-
# @Time     : 2018/11/18 17:34
# @Author   : HuangYin
# @FileName : Compute_Cost_with_Regularization.py
# @Software : PyCharm
import numpy as np


def compute_cost_with_regularization(AL, Y, parameters, lambd):
    """
      Implement the cost function with L2 regularization. See formula (2) above.

      Arguments:
      A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
      Y -- "true" labels vector, of shape (output size, number of examples)
      parameters -- python dictionary containing parameters of the model

      Returns:
      cost - value of the regularized loss function (formula (2))
      """
    m = Y.shape[1]
    L = len(parameters) // 2
    from NN_Deep.Compute_Cost import comput_cost
    cross_entropy_cost = comput_cost(AL, Y)
    L2_regularization_costs = 0.000000000000
    for l in range(1, L + 1):
        L2_regularization_costs += np.sum(np.square(parameters["W" + str(l)]))
    cost = cross_entropy_cost + L2_regularization_costs/(2*m)
    return cost
