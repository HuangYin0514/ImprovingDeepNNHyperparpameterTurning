# -*- coding: utf-8 -*-
# @Time     : 2018/11/18 14:08
# @Author   : HuangYin
# @FileName : Main.py
# @Software : PyCharm

import numpy as np
import matplotlib.pyplot  as plt
import sklearn
import sklearn.datasets
from init_utils import load_dataset

if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
    # plt.rcParams['image.interpolation'] = 'nearest'
    # plt.rcParams['image.cmap'] = 'gray'

    # load data
    train_X, train_Y, test_X, test_Y = load_dataset(True)

    # initialization parameters
    from Initialize_parameters import initialize_parameters_zeros
    from Initialize_parameters import initialize_parameters_random
    from Initialize_parameters import initialize_parameters_he

    parameters = initialize_parameters_he([2, 4, 1])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    from Model import model
    from NN_Deep.Predict import predict

    parameters = model(train_X, train_Y, initialization="he", num_iterations=15000)
    print("On the train set:")
    prediction_train = predict(train_X, train_Y, parameters, print_accuracy=True)
    print("predictions_train : {}".format(prediction_train))
    print("On the test set:")
    prediction_test = predict(test_X, test_Y, parameters, print_accuracy=True)
    print("predictions_test : {}".format(prediction_test))

    # plot decision boundary
    from NN_Deep.Plot_Decision_Boundary import plot_decision_boundary

    plt.figure()
    plt.title("Model with Zeros initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict(x.T, None, parameters), train_X, train_Y)


    plt.show()
