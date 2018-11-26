# -*- coding: utf-8 -*-
# @Time     : 2018/11/26 9:26
# @Author   : HuangYin
# @FileName : Model.py
# @Software : PyCharm

import numpy as np
import matplotlib.pyplot as plt
from Initialize_Parameters_Deep import initialize_parameters_deep
from Compute_Cost import comput_cost
from L_Model_Backward import L_model_Backward
from Update_Parameters import update_parameters
from L_Model_Forward import L_model_forward
from Update_Parameters_wtih_GD import update_Parameters_wtih_GD
from Update_parameters_with_adam import update_parameters_with_adam
from Update_Parameters_with_Momentum import update_parameters_with_momentum
from Initialize_adam import initialize_adam
from Initialize_Velocity import initialize_velocity
from Random_Mini_Batches import random_mini_batches
import opt_utils

def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999,
          epsilon=1e-8, num_epoches=10000, print_cost=True):
    """
       3-layer neural network model which can be run in different optimizer modes.

       Arguments:
       X -- input data, of shape (2, number of examples)
       Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
       layers_dims -- python list, containing the size of each layer
       learning_rate -- the learning rate, scalar.
       mini_batch_size -- the size of a mini batch
       beta -- Momentum hyperparameter
       beta1 -- Exponential decay hyperparameter for the past gradients estimates
       beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
       epsilon -- hyperparameter preventing division by zero in Adam updates
       num_epochs -- number of epochs
       print_cost -- True to print the cost every 1000 epochs

       Returns:
       parameters -- python dictionary containing your updated parameters
       """
    np.random.seed(1)
    costs = []
    seed = 10
    t = 0


    parameters = opt_utils.initialize_parameters(layers_dims)

    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    for i in range(0, num_epoches):

        seed += 1
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed)

        for mini_batche in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batche
            AL, caches = opt_utils.forward_propagation(mini_batch_X, parameters)
            cost = opt_utils.compute_cost(AL, mini_batch_Y)
            grads = opt_utils.backward_propagation(AL, mini_batch_Y, caches)
            if optimizer == "gd":
                parameters = update_Parameters_wtih_GD(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t += 1  # adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,
                                                               epsilon)
        if print_cost and i % 1000 == 0:
            print("Cost after epoch {}:{}".format(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot a figure
    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.xlabel("iterations (per tens)")
    plt.ylabel("cost")
    plt.title("Learning rate = {}".format(learning_rate))

    return parameters
