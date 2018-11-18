# -*- coding: utf-8 -*-
# @Time     : 2018/11/11 17:48
# @Author   : HuangYin
# @FileName : Print_Mislabeled_images.py
# @Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt


def print_mislabel_images(classes, X, y, p):
    """
    :param classes: label
    :param X: testX
    :param y: testY
    :param p: pre_test
    """
    a = p + y
    np
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(10, num_images/5, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + "\n Class: " + classes[y[0, index]].decode(
                "utf-8"))
