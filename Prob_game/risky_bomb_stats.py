
# coding: utf-8

class Risky_bomb_stats(object):
    
    """
    This is a docstring for the Risky_bomb_stats class.
    The class provides statistics about the Risky bomb class objects. 
    count - How many number can be selected during the game.
    probs - Type of the probability density function
    iteration  - Number of times the game is played
    change_bomb - Change the place of the bomb by rounds
    m_iteration- Number of payment matrix created 
    """
    
    def __init__(self,
                 count, 
                 probs, 
                 iteration, 
                 change_bomb, 
                 m_iteration,
                 type_ = None,
                 **kwargs):
        
        self.count = count
        base = Risky_bomb(count = count,
                          probs = probs, 
                          a = kwargs.get('a'),
                          peak = kwargs.get('peak'),
                          smooth = kwargs.get('smooth'),
                          n =  kwargs.get('n'),
                          po =  kwargs.get('po'),
                          df =  kwargs.get('df'))
        
        self.p = base.p
        self.cp = np.cumsum(self.p)
        
        if type_ == "Normal":
            self.payment_m, self.bomb_m = base.payment_matrix(iteration = iteration,
                                                              change_bomb = change_bomb)
        if type_ == "Average":
            self.opt_p_matrix, self.opt_b_matrix = base.avg_payment_matrix(iteration = iteration,
                                                                           m_iteration = m_iteration,
                                                                           change_bomb = change_bomb)
        else:
            self.payment_m, self.bomb_m = base.payment_matrix(iteration = iteration,
                                                              change_bomb = change_bomb)
            self.opt_p_matrix, self.opt_b_matrix = base.avg_payment_matrix(iteration = iteration,
                                                                           m_iteration = m_iteration,
                                                                           change_bomb = change_bomb)
    
    def payment_plot(self):
        """
        Plots the expected payment depending on the guesses, based on only one simulation. 
        """   
        matrix = self.payment_m

        opt_guess = matrix.mean(axis = 1).idxmax()
        opt_payment = matrix.mean(axis = 1).max()
        fig = plt.errorbar(range(1, self.count + 1),matrix.mean(axis = 1), matrix.std(axis = 1), 
                           linestyle = 'None', marker = 'o', color = '#cc3434', ecolor = '#beccea', markersize = 3, linewidth = 2)
        plt.plot(opt_guess + 1, opt_payment, 'D' , markersize = 5, color = '#001871')

        plt.xlabel('Guess', fontsize = 15)
        plt.ylabel('Expected payment', fontsize = 15)
        plt.title('Expexted payment depending on the guess \n Optimal guess: %d \n  Average Payment: %d ' % (opt_guess,opt_payment),
                  fontsize = 20)
        plt.rcParams["figure.figsize"] = [10,7]

        plt.show()    
        
    def avg_payment_plot(self):
        """
        Plots the expected payment depending on the guesses. Average of more simulations. 
        """     
        matrix =  self.opt_p_matrix

        opt_guess = matrix.mean().idxmax()
        opt_payment = matrix.mean().max()
        fig = plt.errorbar(range(1, self.count + 1),matrix.mean(), matrix.std(),
                           linestyle = 'None', marker = 'o', color = '#cc3434', ecolor = '#beccea', markersize = 3, linewidth = 2)
        plt.plot(opt_guess + 1, opt_payment, 'D' , markersize = 5, color = '#001871')

        plt.xlabel('Guess', fontsize = 15)
        plt.ylabel('Expected payment', fontsize = 15)
        plt.title('Expexted payment depending on the guess \n Optimal guess: %d \n  Average Payment: %d ' % (opt_guess,opt_payment),
                  fontsize = 20)
        plt.rcParams["figure.figsize"] = [10,7]

        plt.show()
    
    def theoritical_opt_plot(self):
        """
        Plots the theoritical payment depending on the guesses. Based on the assigned probabilities. 
        """  
        opt = np.asanyarray([x * y for x,y in zip((1-self.cp), range(0,self.count))])
        opt_guess = np.where(opt == opt.max())[0][0]
        opt_payment = float(opt.max())
        
        fig = plt.plot(range(1, self.count + 1), opt, color = '#cc3434', linewidth = 2)
        plt.plot(opt_guess + 1, opt_payment, 'D' , markersize = 5, color = '#001871')

        plt.xlabel('Guess', fontsize = 15)
        plt.ylabel('Expected payment', fontsize = 15)
        plt.title('Expexted payment depending on the guess, based on the probability density \n Optimal guess: %d \n  Expected Payment: %d '
                  %(opt_guess,opt_payment), fontsize = 20)
        plt.rcParams["figure.figsize"] = [10,7]

        plt.show() 
    
    def plot_all(self):
        """
        Plots all graphs: payment_plot, avg_payment_plot, theoritical_opt_plot
        """ 
        matrix = self.payment_m

        # Normal
        opt_guess_n = matrix.mean(axis = 1).idxmax()
        opt_payment = matrix.mean(axis = 1).max()
        fig_n = plt.plot(range(1 ,self.count + 1),matrix.mean(axis = 1),
                         color = '#cc3434', linewidth = 2, alpha = 0.5)
        marker_n = plt.plot(opt_guess_n + 1, opt_payment, '*' , markersize = 8, color = '#cc3434')

        # Avg
        matrix =  self.opt_p_matrix
        
        opt_guess_a = matrix.mean().idxmax()
        opt_payment = matrix.mean().max()
        fig_a = plt.plot(range(1, self.count + 1), matrix.mean(),
                         color = '#001871', linewidth = 2, alpha = 0.5)
        marker_a = plt.plot(opt_guess_a + 1, opt_payment, 'o' , markersize = 8, color = '#001871')

        #Theoritical
        opt = np.asanyarray([x * y for x,y in zip((1-self.cp), range(0,self.count))])
        opt_guess_t = np.where(opt == opt.max())[0][0]
        opt_payment = float(opt.max())

        fig_t = plt.plot(range(1, self.count + 1), opt,
                         color = '#69140e', linewidth = 2)
        marker_t = plt.plot(opt_guess_t + 1, opt_payment ,'D' , markersize = 8, color = '#69140e')


        plt.legend((fig_n[0], marker_n[0], fig_a[0], marker_a[0], fig_t[0], marker_t[0]),
                   ("Empirical","Epirical opt (%d)" %(opt_guess_n),
                    "Avg Empirical","Avg Empirical opt (%d)" %(opt_guess_a),
                    "Theoritical","Theoritical opt (%d)" %(opt_guess_t)))

        plt.xlabel('Guess', fontsize = 15)
        plt.ylabel('Expected payment', fontsize = 15)
        plt.title('Expexted payment depending on the guess by the different probality assignments' , fontsize = 20)
        plt.rcParams["figure.figsize"] = [10,7]

        plt.show()   
        
    def optimum(self, type_o = "Theoritical"):
        """
        Returns the optimal guess by the types.
        type_o - Type of optimum: Empirical, Avg Empirical, Theoritical
        """
        if type_o == "Empirical":
            matrix = self.payment_m
            opt_guess = matrix.mean(axis = 1).idxmax()
        if type_o == "Avg Empirical":
            matrix =  self.opt_p_matrix  
            opt_guess = matrix.mean().idxmax()
        if type_o == "Theoritical":
            opt = np.asanyarray([x * y for x,y in zip((1-self.cp),range(0,self.count))])
            opt_guess = np.where(opt == opt.max())[0][0]
        else:
            raise ValueError("Parameter type_o can only take values: Empirical, Avg Empirical, Theoritical")
        return(opt_guess)
    
    
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import warnings
import spicy
from scipy import stats
from Prob_game.risky_bomb import *

