
# coding: utf-8

from Prob_game.prob_stats import *

class Risky_bomb(Prob_stats):
    """
    This is a docstring for the Risky_bomb class.
    The class calculate the payment for a given guess, plus enable to examine a given guess on long run.
    count - How many number can be selected during the game
    probs - Type of the probability density function
    """
    
    def __init__(self,
                 count,
                 probs,
                 **kwargs):
        
        Prob_stats.__init__(self,
                            count,
                            probs,
                            **kwargs)
    
        self.bomb = int(np.random.choice(range(1, self.count + 1), 1,  p = self.p))   
    
    def change_bomb(self):
        """
        Enables to assign new place of the bomb based on the probability density function.
        """ 
        self.bomb = int(np.random.choice(range(1, self.count + 1), 1,  p = self.p))
        return(self.bomb)
                
    def payment(self, guess, change_bomb= 'False'):
        """
        Determines the payment based on the place of the bomb.
        Guess - Your guess
        change_bomb - Change the place of the bomb by rounds
        """  
        if change_bomb == "T":  
            self.change_bomb()
        self.guess = guess
        if self.count < self.guess or self.guess < 0: 
            raise ValueError("Guess must be smaller than count and greater than 0")
        else:
            if self.bomb > self.guess:
                return(self.guess)
            else:
                return(0) 

        
    def payment_iteration(self, 
                          guess, 
                          iteration, 
                          change_bomb = "T"):
        """
        Determines the payment for more iteration for a given guess. 
        Guess - Your guess
        iteration  - Number of times the game is played
        change_bomb - Change the place of the bomb by rounds
        """  
        self.guess = guess
        
        payment_i = []
        bomb_i =[]
        if change_bomb == "T":  
            for i in range(0, iteration):
                self.change_bomb()
                payment_i.append(self.payment(guess = self.guess))
                bomb_i.append(self.bomb)
        else:
            for i in range(0, iteration):
                payment_i.append(self.payment(guess = self.guess))
                bomb_i.append(self.bomb)
        return(payment_i, bomb_i) 
    
    def payment_matrix(self, iteration, change_bomb = "T"):
        """
        Determines the payment for more iteration for all possible guesses.
        iteration  - Number of times the game is played
        change_bomb - Change the place of the bomb by rounds
        """  
        
        payment_matrix = []
        bomb_matrix = []
        for i in range(1, self.count + 1):
            p, b = self.payment_iteration(iteration = iteration, guess = i, change_bomb = change_bomb)
            payment_matrix.append(p)
            bomb_matrix.append(b)

        payment_matrix = pd.DataFrame(payment_matrix)
        bomb_matrix = pd.DataFrame(bomb_matrix)
        return(payment_matrix,bomb_matrix) 
    
    def avg_payment_matrix(self, 
                           iteration,
                           m_iteration = 10,
                           change_bomb = "T"):
        """
        Determines the payment for more iteration for all possible guesses, based on multiple empircial matrixes.
        iteration  - Number of times the game is played
        change_bomb - Change the place of the bomb by rounds
        m_iteration- Number of payment matrix created
        """  
        
        opt_p_matrix = []
        opt_b_matrix = []
        for i in range(0, m_iteration):
            p, b = self.payment_matrix(iteration = iteration, change_bomb = change_bomb)
            opt_p_matrix.append(p.mean(axis = 1))
            opt_b_matrix.append(b.mean(axis = 1))
            
        opt_p_matrix = pd.DataFrame(opt_p_matrix)
        opt_b_matrix = pd.DataFrame(opt_b_matrix)
        return(opt_p_matrix, opt_b_matrix)
    
    
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import warnings
import spicy
from scipy import stats
from Prob_game.prob_stats import *
