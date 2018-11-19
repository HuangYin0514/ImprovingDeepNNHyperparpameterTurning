# -*- coding: utf-8 -*-
# @Time     : 2018/11/18 14:30
# @Author   : HuangYin
# @FileName : Model.py
# @Software : PyCharm

from L_Model_Forward_with_Dropout import L_model_forward_with_dropout
from L_Model_Backward_with_Dropout import L_model_Backward_with_dropout
from L_Model_Backward_with_Regularization import L_model_Backward_with_regularization
from Compute_Cost_with_Regularization import compute_cost_with_regularization


def model(X, Y, learning_rate=.3, num_iterations=30000, print_cost=True, initialization="he", lambd=0, keep_prob=1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.

    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """
    layers_dims = [X.shape[0], 20, 3, 1]
    costs = []
    grads={}

    from Initialize_parameters import initialize_parameters_zeros
    from Initialize_parameters import initialize_parameters_random
    from Initialize_parameters import initialize_parameters_he

    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    from NN_Deep.L_Model_Forward import L_model_forward
    from NN_Deep.L_Model_Backward import L_model_Backward
    from NN_Deep.Compute_Cost import comput_cost
    from NN_Deep.Update_Parameters import update_parameters

    for i in range(0, num_iterations):
        # it is possible to use both L2 regularization and dropout,
        # but this assignment will only explore one at a time
        assert (lambd == 0 or keep_prob == 1)

        if keep_prob == 1:
            AL, caches = L_model_forward(X, parameters)
        elif keep_prob < 1:
            AL, caches = L_model_forward_with_dropout(X, parameters, keep_prob)

        if lambd == 0:
            cost = comput_cost(AL, Y)
        else:
            cost = compute_cost_with_regularization(AL, Y, parameters, lambd)


        if lambd == 0 and keep_prob == 1:
            grads = L_model_Backward(AL, Y, caches)
        elif lambd != 0:
            grads = L_model_Backward_with_regularization(AL, Y, caches, lambd)
        elif keep_prob < 1:
            grads = L_model_Backward_with_dropout(X, Y, caches, keep_prob)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost == True and i % 10000 == 0:
            print("Cost after iteration {} : {}".format(i, cost))
            costs.append(cost)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("iterations of hundreds")
    plt.title("Learning rate = {}".format(str(learning_rate)))

    return parameters
