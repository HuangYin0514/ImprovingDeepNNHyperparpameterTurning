# -*- coding: utf-8 -*-
# @Time     : 2018/11/18 15:06
# @Author   : HuangYin
# @FileName : Initialize_parameters_zeros.py
# @Software : PyCharm

import numpy as np


def initialize_parameters_zeros(layers_dim):
    """
      Arguments:
      layer_dims -- python array (list) containing the size of each layer.

      Returns:
      parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                      W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                      b1 -- bias vector of shape (layers_dims[1], 1)
                      ...
                      WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                      bL -- bias vector of shape (layers_dims[L], 1)
      """
    parameters = {}
    L = len(layers_dim)
    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dim[l], layers_dim[l - 1]))
        parameters['b' + str(l)] = np.zeros((layers_dim[l], 1))
    assert (parameters["W1"].shape == (layers_dim[1], layers_dim[0]))
    assert (parameters["b1"].shape == (layers_dim[1], 1))
    return parameters


def initialize_parameters_random(layers_dim):
    """
      Arguments:
      layer_dims -- python array (list) containing the size of each layer.

      Returns:
      parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                      W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                      b1 -- bias vector of shape (layers_dims[1], 1)
                      ...
                      WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                      bL -- bias vector of shape (layers_dims[L], 1)
      """
    parameters = {}
    L = len(layers_dim)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dim[l], layers_dim[l - 1])*1.5
        parameters['b' + str(l)] = np.zeros((layers_dim[l], 1))
    assert (parameters["W1"].shape == (layers_dim[1], layers_dim[0]))
    assert (parameters["b1"].shape == (layers_dim[1], 1))
    return parameters


def initialize_parameters_he(layers_dim):
    """
      Arguments:
      layer_dims -- python array (list) containing the size of each layer.

      Returns:
      parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                      W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                      b1 -- bias vector of shape (layers_dims[1], 1)
                      ...
                      WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                      bL -- bias vector of shape (layers_dims[L], 1)
      """
    parameters = {}
    L = len(layers_dim)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dim[l], layers_dim[l - 1]) * np.sqrt(2 / layers_dim[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dim[l], 1))
    assert (parameters["W1"].shape == (layers_dim[1], layers_dim[0]))
    assert (parameters["b1"].shape == (layers_dim[1], 1))
    return parameters
