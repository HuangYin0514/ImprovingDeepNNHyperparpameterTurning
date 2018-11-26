# -*- coding: utf-8 -*-
# @Time     : 2018/11/24 8:13
# @Author   : HuangYin
# @FileName : Initialize_Velocity.py
# @Software : PyCharm

import numpy as np


def initialize_velocity(parameters):
    """
        Initializes the velocity as a python dictionary with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL"
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        Arguments:
        parameters -- python dictionary containing your parameters.
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl

        Returns:
        v -- python dictionary containing the current velocity.
                        v['dW' + str(l)] = velocity of dWl
                        v['db' + str(l)] = velocity of dbl
        """
    L = len(parameters) // 2
    v = {}
    for l in range(0, L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return v
