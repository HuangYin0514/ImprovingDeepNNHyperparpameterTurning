# -*- coding: utf-8 -*-
# @Time     : 2018/11/19 14:30
# @Author   : HuangYin
# @FileName : GradientCheck.py
# @Software : PyCharm

from ForwardPropagation import forward_propagation
from BackwardPropagation import backward_propagation
import numpy as np

def gradient_check(x, theta, epsilon=1e-7):
    """
      Implement the backward propagation presented in Figure 1.

      Arguments:
      x -- a real-valued input
      theta -- our parameter, a real number as well
      epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

      Returns:
      difference -- difference (2) between the approximated gradient and the backward propagation gradient
      """
    thetaplus = theta + epsilon
    thetaminus = theta - epsilon
    J_plus = forward_propagation(x, thetaplus)
    J_minus = forward_propagation(x, thetaminus)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)

    grad = backward_propagation(x, theta)

    enumerate = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = enumerate / denominator

    if difference < 1e-7:
        print("The gradient is correct!")
    else:
        print("The gradient is wrong!")

    return difference
