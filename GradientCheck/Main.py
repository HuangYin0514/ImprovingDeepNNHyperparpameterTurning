# -*- coding: utf-8 -*-
# @Time     : 2018/11/19 14:10
# @Author   : HuangYin
# @FileName : Main.py
# @Software : PyCharm

from ForwardPropagation import forward_propagation
from BackwardPropagation import backward_propagation
from Gradient_Check import gradient_check

if __name__ == '__main__':
    x, theta = 2, 4
    J = forward_propagation(x, theta)
    print("J = {}".format(J))
    dtheta = backward_propagation(x, theta)
    print("dtheta = {}".format(dtheta))
    difference = gradient_check(x, theta)
    print("difference = {}".format(difference))
