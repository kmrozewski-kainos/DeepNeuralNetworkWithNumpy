# -*- coding: utf-8 -*-
from deep_neural_network.activation import relu, sigmoid

class ActivationObject(object):
    
    def __init__(self):
        self.sigmoid = sigmoid.Sigmoid()
        self.relu = relu.ReLU()
