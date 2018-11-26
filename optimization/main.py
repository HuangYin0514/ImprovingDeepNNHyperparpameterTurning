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
import opt_utils

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
# mini_batches(mini_batche(mini_batch_X, mini_batch_Y), mini_batche(mini_batch_X, mini_batch_Y), ...)
mini_batches = random_mini_batches(X, Y, mini_batch_size)
print("shape of the 1st mini batch_X: " + str(mini_batches[0][0].shape))
print("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
print("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
print("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
print("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
print("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
print("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

# test initialize_velocity
from Initialize_Velocity import initialize_velocity

parameters = testCase.initialize_velocity_test_case()
v = initialize_velocity(parameters)
print("v[\"dW1\"] = {}".format(str(v["dW1"])))
print("v[\"db1\"] = {}".format(str(v["db1"])))
print("v[\"dW2\"] = {}".format(str(v["dW2"])))
print("v[\"db2\"] = {}".format(str(v["db2"])))

# test update_parameters_with_momentum
from Update_Parameters_with_Momentum import update_parameters_with_momentum

parameters, grads, v = testCase.update_parameters_with_momentum_test_case()
update_parameters_with_momentum(parameters, grads, v, beta=0.9, learning_rate=0.01)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print("v[\"dW1\"] = " + str(v["dW1"]))
print("v[\"db1\"] = " + str(v["db1"]))
print("v[\"dW2\"] = " + str(v["dW2"]))
print("v[\"db2\"] = " + str(v["db2"]))

# test initialize_adam
from Initialize_adam import initialize_adam

parameters = testCase.initialize_adam_test_case()
v, s = initialize_adam(parameters)
print("v[\"dW1\"] = " + str(v["dW1"]))
print("v[\"db1\"] = " + str(v["db1"]))
print("v[\"dW2\"] = " + str(v["dW2"]))
print("v[\"db2\"] = " + str(v["db2"]))
print("s[\"dW1\"] = " + str(s["dW1"]))
print("s[\"db1\"] = " + str(s["db1"]))
print("s[\"dW2\"] = " + str(s["dW2"]))
print("s[\"db2\"] = " + str(s["db2"]))

# test update adam
from Update_parameters_with_adam import update_parameters_with_adam

parameters, grads, v, s = testCase.update_parameters_with_adam_test_case()
parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t=2)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print("v[\"dW1\"] = " + str(v["dW1"]))
print("v[\"db1\"] = " + str(v["db1"]))
print("v[\"dW2\"] = " + str(v["dW2"]))
print("v[\"db2\"] = " + str(v["db2"]))
print("s[\"dW1\"] = " + str(s["dW1"]))
print("s[\"db1\"] = " + str(s["db1"]))
print("s[\"dW2\"] = " + str(s["dW2"]))
print("s[\"db2\"] = " + str(s["db2"]))

# test model
from optimization.Model import model
from Predict import predict
from Plot_Decision_Boundary import plot_decision_boundary

train_X, train_Y = opt_utils.load_dataset(True)
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims=layers_dims, optimizer="adam")
predictions = predict(train_X, train_Y, parameters,print_accuracy=True)
# plot decision boundary
plt.figure()
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
plot_decision_boundary(lambda x: predict(x.T, train_Y, parameters), train_X, train_Y)
plt.show()
