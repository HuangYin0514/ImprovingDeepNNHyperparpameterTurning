# -*- coding: utf-8 -*-
# @Time     : 2018/11/2 21:51
# @Author   : HuangYin
# @FileName : Sigmoid.py
# @Software : PyCharm
import numpy as np


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    cache = z
    return a,cache
