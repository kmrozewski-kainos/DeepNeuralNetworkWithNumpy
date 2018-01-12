#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from deep_neural_network import model
from datasets import Train, Test

######## Hyperparameters
layers_dims = [12288, 20, 7, 5, 1]

######## Load data
train = Train().dataset
test = Test().dataset

X_train, Y_train = train['x'], train['y']
X_test, Y_test, classes = test['x'], test['y'], test['classes']

# X_train is in 3 dimensions (209 observations, 64x64 pixels in 3 channels - RGB)
print X_train.shape

######### Flattening input data

# output shape should be 209 x 12288 which is (64 * 64 * 3)
# also 12288 is the numper of input layer units (A0)
# then transpose the X_train matrix
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3]).T
# transpose Y_train matrix as wells
Y_train = Y_train.reshape(1, Y_train.shape[0])

print X_train.shape
print Y_train.shape


# Initialize neural network
nn_model = model.Model(layers_dims)

# train neural net
nn_model.L_layer_model(X_train, Y_train)