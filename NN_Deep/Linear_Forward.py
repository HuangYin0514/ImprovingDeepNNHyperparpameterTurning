# -*- coding: utf-8 -*-
# @Time     : 2018/11/2 21:35
# @Author   : HuangYin
# @FileName : Linear_Forward_with_Regularization.py
# @Software : PyCharm
import numpy as np


def linear_forward(A, W, b):
    """
       Implement the linear part of a layer's forward propagation.

       Arguments:
       A -- activations from previous layer (or input data): (size of previous layer, number of examples)
       W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
       b -- bias vector, numpy array of shape (size of the current layer, 1)

       Returns:
       Z -- the input of the activation function, also called pre-activation parameter
       cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
       """
    # Z = np.dot(W, A) + b
    Z = W.dot(A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache
