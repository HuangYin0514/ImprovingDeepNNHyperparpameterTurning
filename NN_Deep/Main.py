# -*- coding: utf-8 -*-
# @Time     : 2018/11/4 14:44
# @Author   : HuangYin
# @FileName : Main.py
# @Software : PyCharm

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy
import skimage

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# load data
from lr_utils import load_dataset

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

# show data
index = 10
plt.imshow(train_set_x_orig[index])
print("y  = " + str(train_set_y_orig[0, index]) + " It`s a " + str(
    classes[train_set_y_orig[0, index]].decode("utf-8")) + " picture")

# look num information of data set
m_train = train_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
m_test = test_set_x_orig.shape[0]
print("Number of train example : " + str(m_train))
print("Number of test example : " + str(m_test))
print("Each image is of size :( " + str(num_px) + "," + str(num_px) + ",3)")
print("train_x_orig shape: " + str(train_set_x_orig.shape))
print("train_y_orig shape: " + str(train_set_y_orig.shape))
print("test_x_orig shape: " + str(test_set_x_orig.shape))
print("test_y_orig shape: " + str(test_set_y_orig.shape))

# reshape the training and test example
train_x_flatten = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T
test_x_flatten = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T

# Standardize data to have feature values between 0 and 1
train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

print("train_x`s shape " + str(train_x.shape))
print("test_x`s shape " + str(test_x.shape))

# constants defining the model
n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

# two Layer_Model
from Two_Layer_Model import two_layer_model
from Predict import predict

parameters = two_layer_model(train_x, train_set_y_orig, layers_dims, num_iterations=0, print_cost=True)
prediction_train = predict(train_x, train_set_y_orig, parameters)
accuracy_train = (1 - np.mean(np.abs(prediction_train - train_set_y_orig))) * 100
print("Accuracy of train = {}%".format(accuracy_train))
prediction_test = predict(test_x, test_set_y_orig, parameters)
accuracy_test = (1 - np.mean(np.abs(prediction_test - test_set_y_orig))) * 100
print("Accuracy of test = {}%".format(accuracy_test))

# 5 Layer_Model
from L_Layer_Model import L_layer_model

layers_dims = [12288, 20, 7, 5, 1]
parameters = L_layer_model(train_x, train_set_y_orig, layers_dims, num_iterations=2500, print_cost=True)
prediction_train = predict(train_x, train_set_y_orig, parameters)
accuracy_train = (1 - np.mean(np.abs(prediction_train - train_set_y_orig))) * 100
print("Accuracy of train = {}%".format(accuracy_train))
prediction_test = predict(test_x, test_set_y_orig, parameters)
accuracy_test = (1 - np.mean(np.abs(prediction_test - test_set_y_orig))) * 100
print("Accuracy of test = {}%".format(accuracy_test))

from Print_Mislabeled_images import print_mislabel_images

print_mislabel_images(classes, test_x, test_set_y_orig, prediction_test)

# my pic test
my_image = "mytest.png"
my_label_y = [1]
my_label_y = np.array(my_label_y).reshape(1,1)
fname = "MyImageSet/" + my_image
image = np.array(plt.imread(fname))
my_image = np.array(skimage.transform.resize(image, output_shape=(num_px, num_px, 3))).reshape((num_px * num_px * 3, 1))

my_predicted_image = predict(my_image, my_label_y, parameters)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[
    int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
plt.show()

