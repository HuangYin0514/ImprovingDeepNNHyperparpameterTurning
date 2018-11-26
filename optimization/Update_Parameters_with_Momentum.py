# -*- coding: utf-8 -*-
# @Time     : 2018/11/24 8:28
# @Author   : HuangYin
# @FileName : Update_Parameters_with_Momentum.py
# @Software : PyCharm

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
        Update parameters using Momentum

        Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients for each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        v -- python dictionary containing the current velocity:
                        v['dW' + str(l)] = ...
                        v['db' + str(l)] = ...
        beta -- the momentum hyperparameter, scalar
        learning_rate -- the learning rate, scalar

        Returns:
        parameters -- python dictionary containing your updated parameters
        v -- python dictionary containing your updated velocities
        """
    L = len(parameters) // 2

    for l in range(0, L):
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]
        parameters["W" + str(l + 1)] -= learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * v["db" + str(l + 1)]

    return parameters, v
