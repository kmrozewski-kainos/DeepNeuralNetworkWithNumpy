# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from deep_neural_network.parameters import Parameters
from deep_neural_network.forward_propagation import Forward_Propagation
from deep_neural_network.back_propagation import Back_Propagation

class Model:

    def __init__(self, layer_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
        # Initialize model hyperparameters
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost

        # Initialize modules
        self.parameters = Parameters(self.layer_dims)
        self.forward_propagation = Forward_Propagation()
        self.back_propagation = Back_Propagation()


    def L_layer_model(self, X, Y):#lr was 0.009
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps

        Returns:
        model_parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)
        costs = []                         # keep track of cost

        # Parameters initialization.
        ### START CODE HERE ###
        model_parameters = self.parameters.initialize(self.layer_dims)
        ### END CODE HERE ###

        # Loop (gradient descent)
        for i in range(0, self.num_iterations):
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = self.forward_propagation.L_model_forward(X, model_parameters)

            # Compute cost.
            cost = self.parameters.compute_cost(AL, Y)

            # Backward propagation.
            grads = self.back_propagation.L_model_backward(AL, Y, caches)

            # Update parameters.
            model_parameters = self.parameters.update_parameters(model_parameters, grads, self.learning_rate)

            # Print the cost every 100 training example
            if self.print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if self.print_cost and i % 100 == 0:
                costs.append(cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()

        return self.parameters