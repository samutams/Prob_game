
# coding: utf-8

class Prob_stats(object):
    
    """
    This is a docstring for the Prob_stats class.
    The aim of this class is to provide statistics and plots
    about the created probabiliteis within the Probs class.
    count - How many number can be selected during the game.
    probs - Type of the probability density function
    """
    
    def __init__(self,
                 count,
                 probs = "equal",
                 **kwargs):
        
        self.count = count
        self.probs = probs
        
        prob_base = Prob(count = self.count)
        
        if self.probs == "equal":
            self.p = prob_base.equal()
            self.arg = {"Count": self.count}
        if self.probs == "fixed":
            if kwargs.get('peak') is None: 
                raise ValueError("Argument peak must be specified as the place of the bomb")
            else:
                self.peak = kwargs.get('peak') 
                self.p = prob_base.fixed(peak = self.peak)
                self.arg = {"Count": self.count, "Peak" : self.peak}
        if self.probs == "random_skew":
            if kwargs.get('peak')  is None or kwargs.get('smooth')  is None: 
                raise ValueError("Argument peak and smooth must be specified")
            else:
                self.smooth = kwargs.get('smooth')
                self.peak = kwargs.get('peak') 
                if self.smooth == "True": 
                    self.p = prob_base.random_skew(smooth = "True", peak = self.peak)
                else:
                    self.p = prob_base.random_skew(smooth = "False", peak = self.peak)
                self.arg = {"Count": self.count, "Smooth" : self.smooth, "Peak" : self.peak}
        if self.probs == "decreasing":
            self.p = prob_base.decreasing()
            self.arg = {"Count": self.count}
        if self.probs == "increasing":
            self.p = prob_base.increasing()
            self.arg = {"Count": self.count}
        if self.probs == "binom":
            if kwargs.get('n') is None or kwargs.get('po') is None:
                raise ValueError("Argument n and po must be specified for binominal distribution")
            else:
                self.n = kwargs.get('n')
                self.po = kwargs.get('po')
                self.p = prob_base.binom(n = self.n , po = self.po)
                self.arg = {"Count": self.count, "N" : self.n, "Po" : self.po}
        if self.probs == "skewnorm":
            if kwargs.get('a') is None:
                raise ValueError("Argument a must be specified for this distribution")
            else:
                self.a = kwargs.get('a')
                self.p = prob_base.skewnorm(a = self.a)
                self.arg = {"Count": self.count, "A" : self.a}
        if self.probs == "t":
            if kwargs.get('df') is None:
                raise ValueError("Argument df must be specified for t distribution")
            else:
                self.df = kwargs.get('df')
                self.p = prob_base.t(df = self.df)    
                self.arg = {"Count": self.count, "DF" : self.df}
        if self.probs == "alpha":
            if kwargs.get('a') is None:
                raise ValueError("Argument a must be specified for this distribution")
            else:
                self.a = kwargs.get('a')
                self.p = prob_base.alpha(a = self.a) 
                self.arg = {"Count": self.count, "A" : self.a}
        if self.probs == "gamma":
            if kwargs.get('a') is None:
                raise ValueError("Argument a must be specified for this distribution")
            else:
                self.a = kwargs.get('a')
                self.p = prob_base.gamma(a = self.a) 
                self.arg = {"Count": self.count, "A" : self.a}                
        if self.probs == "beta":
            if kwargs.get('a') is None or kwargs.get('b'):
                raise ValueError("Argument a and bmust be specified for this distribution")
            else:
                self.a = kwargs.get('a')
                self.b = kwargs.get('b')
                self.p = prob_base.beta(a = self.a, b = self.b)        
                self.arg = {"Count": self.count, "A" : self.a , "B" : self.b}
                
    def prob_plot(self):
        """
        Plots the probabilities by the different counts
        """   
        plt.plot(range(1, self.count + 1), self.p, color = '#001871')
        
        plt.xlabel('Count', fontsize = 15)
        plt.ylabel('Probability of selection', fontsize = 15)
        plt.title('Probabiliy of selection by the different counts \n Type: %s \n %s' %(self.probs, self.arg), fontsize = 20)
        plt.rcParams["figure.figsize"] = [10,7]

        plt.show() 
        
    def cum_prob_plot(self):
        """
        Plots the cumulative probabilities by the different counts
        """  
        self.cp = np.cumsum(self.p)

        plt.plot(range(1, self.count + 1), self.cp, color = '#001871')

        plt.xlabel('Count', fontsize = 15)
        plt.ylabel('Cumulative probability of selection', fontsize = 15)
        plt.title('Cumulative probabiliy of selection by the different counts \n Type: %s \n %s' %(self.probs, self.arg), fontsize = 20)
        plt.rcParams["figure.figsize"] = [10,7]

        plt.show()
        
    def stats(self):
        """
        Provides summary stats for the probability density function.
        """  
        self.p_stats = stats.describe(self.p)
        return(self.p_stats)
    
    
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import warnings
import spicy
from scipy import stats
from Prob_game.prob import *
