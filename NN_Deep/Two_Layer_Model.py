# -*- coding: utf-8 -*-
# @Time     : 2018/11/4 16:35
# @Author   : HuangYin
# @FileName : Two_Layer_Model.py
# @Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt


def two_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    costs = []
    # initialize parameters
    from Initialize_Parameters_Deep import initialize_parameters_deep
    parameters = initialize_parameters_deep(layer_dims)

    for i in range(num_iterations):

        # forward propagation
        from L_Model_Forward import L_model_forward
        AL, caches = L_model_forward(X, parameters)

        # compute cost
        from Compute_Cost import comput_cost
        cost = comput_cost(AL, Y)
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

        # backward propagation
        from L_Model_Backward import L_model_Backward
        grads = L_model_Backward(AL, Y, caches)

        # update parameters
        from Update_Parameters import update_parameters
        parameters = update_parameters(parameters, grads, learning_rate)

    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.xlabel("iterations (per tens)")
    plt.ylabel("cost")
    plt.title("Learning rate = " + str(learning_rate))

    return parameters
