# -*- coding: utf-8 -*-
# @Time     : 2018/11/4 20:06
# @Author   : HuangYin
# @FileName : L_Layer_Model.py
# @Software : PyCharm

# import NewTest as nt
from Initialize_Parameters_Deep import initialize_parameters_deep
from L_Model_Forward import L_model_forward
from L_Model_Backward import L_model_Backward
from Compute_Cost import comput_cost
from Update_Parameters import update_parameters
import matplotlib.pyplot as plt
import numpy as np



def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    # 改
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):
        AL, caches =L_model_forward(X, parameters)
        cost = comput_cost(AL, Y)
        if print_cost and i % 100 == 0:
            costs.append(cost)
            print("Cost after iterations {}:{}".format(i, cost))
        grads = L_model_Backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
    # plot a figure
    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.xlabel("iterations (per tens)")
    plt.ylabel("cost")
    plt.title("Learning rate = {}".format(learning_rate))

    return parameters
