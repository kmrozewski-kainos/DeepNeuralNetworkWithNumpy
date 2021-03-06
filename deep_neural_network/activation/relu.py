# -*- coding: utf-8 -*-

from deep_neural_network.activation.activation_base import ActivationBase
import numpy as np

class ReLU(ActivationBase):

    def __init__(self):
        pass

    def activation(self, Z):
        """
        Implement the RELU function.
        Arguments:
        Z -- Output of the linear layer, of any shape
        Returns:
        A -- Post-activation parameter, of the same shape as Z
        cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
        """

        A = np.maximum(0,Z)

        assert(A.shape == Z.shape)

        cache = Z
        return A, cache

    def activation_backward(self, dA, cache):
        """
        Implement the backward propagation for a single RELU unit.
        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently
        Returns:
        dZ -- Gradient of the cost with respect to Z
        """

        Z = cache
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.

        # When z <= 0, you should set dz to 0 as well.
        dZ[Z <= 0] = 0

        assert (dZ.shape == Z.shape)

        return dZ
