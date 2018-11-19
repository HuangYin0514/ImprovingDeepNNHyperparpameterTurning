# -*- coding: utf-8 -*-
# @Time     : 2018/11/19 14:21
# @Author   : HuangYin
# @FileName : ForwardPropagation.py
# @Software : PyCharm

import numpy as np


def forward_propagation(x, theta):
    """
       Implement the linear forward propagation (compute J) presented in Figure 1 (J(theta) = theta * x)

       Arguments:
       x -- a real-valued input
       theta -- our parameter, a real number as well

       Returns:
       J -- the value of function J, computed using the formula J(theta) = theta * x
       """
    J = np.dot(theta, x)

    return J
