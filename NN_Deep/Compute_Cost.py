# -*- coding: utf-8 -*-
# @Time     : 2018/11/3 11:10
# @Author   : HuangYin
# @FileName : Compute_Cost.py
# @Software : PyCharm
import numpy as np


def comput_cost(AL, Y):
    """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
    m = Y.shape[1]
    # cost = -1 * (1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - Y)) equal below
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost
