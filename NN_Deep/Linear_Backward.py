# -*- coding: utf-8 -*-
# @Time     : 2018/11/3 11:32
# @Author   : HuangYin
# @FileName : Linear_Backward.py
# @Software : PyCharm
import numpy as np


def linear_backward(dZ, cache):
    """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True)/ m
    dA_prev = np.dot(W.T, dZ)
    # 写法疑问
    # np.squeeze(np.sum(dZ, axis=1, keepdims=True)/ m)
    # 可用写法
    # np.squeeze(np.sum(dZ, axis=1, keepdims=True)/ m)/1
    # db = np.sum(dZ, axis=1, keepdims=True) / m

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    # assert (isinstance(db[0], float))

    return dA_prev, dW, db
