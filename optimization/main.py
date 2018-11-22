# -*- coding: utf-8 -*-
# @Time     : 2018/11/22 8:15
# @Author   : HuangYin
# @FileName : main.py
# @Software : PyCharm

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sklearn
import sklearn.datasets
import math
import testCase

plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots

# gradient decent algorithm
from Update_Parameters_wtih_GD import update_Parameters_wtih_GD

parameters, grads, learning_rate = testCase.update_parameters_with_gd_test_case()
parameters = update_Parameters_wtih_GD(parameters, grads, learning_rate)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# split m to mini batches
from Random_Mini_Batches import random_mini_batches

X, Y, mini_batch_size = testCase.random_mini_batches_test_case()
mini_batches = random_mini_batches(X, Y, mini_batch_size)
# mini_batches(mini_batche(mini_batch_X, mini_batch_Y), mini_batche(mini_batch_X, mini_batch_Y), ...)
print("shape of the 1st mini batch_X: " + str(mini_batches[0][0].shape))
print("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
print("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
print("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
print("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
print("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
print("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))