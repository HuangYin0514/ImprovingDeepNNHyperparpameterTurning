# -*- coding: utf-8 -*-
# @Time     : 2018/11/18 16:51
# @Author   : HuangYin
# @FileName : Main.py
# @Software : PyCharm

import numpy as np
import matplotlib.pyplot  as plt
import sklearn
import sklearn.datasets

if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots

    # load data
    from reg_utils import load_2D_dataset

    train_X, train_Y, test_X, test_Y = load_2D_dataset(True)

    # compute cost
    from regularization.Model import model
    from NN_Deep.Predict import predict

    parameters = model(train_X, train_Y, keep_prob=.86,learning_rate=.3)
    print("Training set with regularization")
    predict(train_X, train_Y, parameters, print_accuracy=True)
    print("test set with regularization")
    predict(test_X, test_Y, parameters, print_accuracy=True)

    # plot decision boundary
    from NN_Deep.Plot_Decision_Boundary import plot_decision_boundary

    plt.figure()
    plt.title("Model with Zeros initialization")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: predict(x.T, None, parameters), train_X, train_Y)

    plt.show()
