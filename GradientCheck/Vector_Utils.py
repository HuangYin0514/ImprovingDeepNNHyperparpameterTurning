# -*- coding: utf-8 -*-
# @Time     : 2018/11/19 15:15
# @Author   : HuangYin
# @FileName : Vector_Utils.py
# @Software : PyCharm

import numpy as np


def dictionary_to_vector(parameters):
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        new_vector = np.reshape(parameters[key], (-1, 1))
        keys += [key] * new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count += 1
    return theta, keys
