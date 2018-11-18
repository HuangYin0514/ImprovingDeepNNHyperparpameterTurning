# -*- coding: utf-8 -*-
# @Time     : 2018/11/4 19:20
# @Author   : HuangYin
# @FileName : Predict.py
# @Software : PyCharm
from L_Model_Forward import L_model_forward
import numpy as np


def predict(X, Y, parameters,print_accuracy = False):
    m = X.shape[1]
    prediction = np.zeros((1, m))
    AL, caches = L_model_forward(X, parameters)
    # for i in range(AL.shape[1]):
    #     # pridiction[0, i] = 1 if AL > 0.5 else 0
    #     if AL[0, i] > 0.5:
    #         pridiction[0,i] = 1
    #     else:
    #         pridiction[0,i] =0
    prediction[AL >= 0.5] = 1
    prediction[AL < 0.5] = 0

    if print_accuracy:
        accuracy = np.mean(prediction == Y)
        print("Accuracy : {}".format(accuracy))

    return prediction
