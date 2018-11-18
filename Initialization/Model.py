# -*- coding: utf-8 -*-
# @Time     : 2018/11/18 14:30
# @Author   : HuangYin
# @FileName : Model.py
# @Software : PyCharm

def model(X, Y, learning_rate=.01, num_iterations=15000, print_cost=True, initialization="he"):
    """
        Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

        Arguments:
        X -- input data, of shape (2, number of examples)
        Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
        learning_rate -- learning rate for gradient descent
        num_iterations -- number of iterations to run gradient descent
        print_cost -- if True, print the cost every 1000 iterations
        initialization -- flag to choose which initialization to use ("zeros","random" or "he")

        Returns:
        parameters -- parameters learnt by the model
        """

    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]
    costs = []

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
        AL, caches = L_model_forward(X, parameters)
        cost = comput_cost(AL, Y)
        grads = L_model_Backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost == True and i % 1000 == 0:
            print("Cost after iteration {} : {}".format(i, cost))
            costs.append(cost)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("iterations of hundreds")
    plt.title("Learning rate = {}".format(str(learning_rate)))

    return parameters
