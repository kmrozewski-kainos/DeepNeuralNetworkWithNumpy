# -*- coding: utf-8 -*-

class ActivationBase(object):
    
    def __init__(self):
        pass
    
    def activation(self, Z):
        raise NotImplementedError()
    
    def activation_backward(self, dA, cache):
        raise NotImplementedError()
