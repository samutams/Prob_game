
# coding: utf-8

class Nn_obj(object):
    """
    This is a docstring for the Nn_obj class.
    The class creates an nn object and enables to perform forward propagation. 
    count - How many number can be selected during the game.
    n_input - Number of input nodes
    n_hidden - Number of hidden nodes
    n_output - Number of output nodes
    add_input - Statistics that should be considered during the nn training
    """
    
    def __init__(self,
                 count = 99, 
                 n_input = 1, 
                 n_hidden = 5, 
                 n_output = 1,
                 add_input = None):
        self.count = count
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output     
        self.guess = np.random.randint(1, self.count)
        self.input_values = [self.guess]
        if add_input != None: 
            inp = [self.guess]
            inp.extend(add_input)
            self.input_values = inp
            
            if self.n_input != len(self.input_values):
                self.n_input = len(self.input_values)
                warnings.warn("n_input was modified to match the number of inputs")
                
        self.w_input = np.random.uniform(-1, 1, (self.n_hidden, self.n_input))
        self.w_output = np.random.uniform(-1, 1, (self.n_output, self.n_hidden))
        
    # NEURAL NETWORK
    def nn(self):
        """
        Runs the forward propagation of an nn algorithm.
        """         
        c = self.count / 2
        norm = lambda x: (x - c) / c # Normalization
        inv_norm = lambda x: x * c + c # Inverse Norm function

        # Normalize
        input_ = np.array(list([norm(x) for x in self.input_values]))
        # SIMPLE MLP
        
        af = lambda x: np.tanh(x)               # activation function
        h1 = af(np.dot(self.w_input, input_ ))  # hidden layer
        out = af(np.dot(self.w_output, h1))          # output layer
        self.guess = int(inv_norm(out))
    
    def weight_update(self, w_input, w_output):
        """
        Updates the weights. 
        w_input - Input weights
        w_output - Output weights
        """   
        self.w_input = w_input
        self.w_output = w_output
        
    def add_input_update(self, add_input_):
        """
        Updates input values. 
        add_input_ - New input values to be added
        """   
        inp = [self.guess]
        inp.extend(add_input_)
        self.input_values = inp
    
    

import numpy as np
import pandas as pd

