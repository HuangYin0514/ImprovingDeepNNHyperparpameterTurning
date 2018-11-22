# -*- coding: utf-8 -*-
# @Time     : 2018/11/22 8:57
# @Author   : HuangYin
# @FileName : Random_Mini_Batches.py
# @Software : PyCharm

import numpy as np
import math


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(seed)
    mini_batches = []
    m = X.shape[1]

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    assert (shuffled_Y.shape == (1, m))

    num_complete_miniBatches = math.floor(m / mini_batch_size)

    for k in range(0, num_complete_miniBatches):
        # 这里的 xx:yy 指的是[xx,yy) 左闭右开区间
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        assert (mini_batch_Y.shape == (1, mini_batch_size))
        mini_batche = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batche)

    # 处理最后不足一个mini_batche_size的部分
    if m % mini_batch_size != 0:
        # num_complete_miniBatches = math.floor(m / mini_batch_size)
        end = m - mini_batch_size * num_complete_miniBatches
        mini_batch_X = shuffled_X[:, num_complete_miniBatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_miniBatches * mini_batch_size:]
        mini_batche = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batche)

    return mini_batches
