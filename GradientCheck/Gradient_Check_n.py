# -*- coding: utf-8 -*-
# @Time     : 2018/11/19 15:24
# @Author   : HuangYin
# @FileName : Gradient_Check_n.py
# @Software : PyCharm

from Vector_Utils import vector_to_dictionary
from Vector_Utils import dictionary_to_vector
from Vector_Utils import gradients_to_vector
import numpy as np
from Forward_Propagation_n import forward_propagation_n
from Backwar_Propagation_n import backward_propagation_n


def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    """
       Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

       Arguments:
       parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
       grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters.
       x -- input datapoint, of shape (input size, 1)
       y -- true "label"
       epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

       Returns:
       difference -- difference (2) between the approximated gradient and the backward propagation gradient
       """
    grad = gradients_to_vector(gradients)
    parameters_values, _ = dictionary_to_vector(parameters)
    num_parameters = parameters_values.shape[0]
    J_Plus = np.zeros((num_parameters, 1))
    J_Minus = np.zeros((num_parameters, 1))
    grad_approx = np.zeros((num_parameters, 1))

    for i in range(0, num_parameters):
        theta_plus = np.copy(parameters_values)
        theta_plus[i][0] += epsilon
        J_Plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_plus))

        theta_minus = np.copy(parameters_values)
        theta_minus[i][0] += epsilon
        J_Minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_minus))

        grad_approx[i] = (J_Plus[i] - J_Minus[i]) / (2 * epsilon)

    enumerate = np.linalg.norm(grad - grad_approx)
    denominator = np.linalg.norm(grad) + np.linalg.norm(grad_approx)
    difference = enumerate / denominator
    if difference > 1e-7:
        print("There is a mistake in the backward propagation! difference = " + str(difference))
    else:
        print("Your backward propagation works perfectly fine! difference = " + str(difference))

    return difference
