# -*- coding: utf-8 -*-
# @Time     : 2018/11/3 20:54
# @Author   : HuangYin
# @FileName : Update_Parameters.py
# @Software : PyCharm

def update_parameters(parameters, grads, learning_rate):
    """
     Update parameters using gradient descent

     Arguments:
     parameters -- python dictionary containing your parameters
     grads -- python dictionary containing your gradients, output of L_model_backward

     Returns:
     parameters -- python dictionary containing your updated parameters
                   parameters["W" + str(l)] = ...
                   parameters["b" + str(l)] = ...
     """
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters
