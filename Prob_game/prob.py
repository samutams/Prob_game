
# coding: utf-8

class Prob(object):
    """
    This is a docstring for the Prob class.
    The aim of this class is to create probability
    density functions for given number of varaibles (set by count). 
    The different functions create distinct probability densities. 
    count - How many number can be selected during the game
    """
    
    def __init__(self, count = 99):
        if count <= 0:
            raise ValueError("Count must be greater than 0")
        else:
            self.count = count

    def equal(self):
        """
        Provides equal probabilities. 
        """
        prob = [1 / self.count] * self.count
        return(prob)
    
    def fixed(self, peak = 1):        
        """
        Fixing the place of 100% prob, and 0% otherwise
        peak - Place of 100% prob
        """
        peak = peak
        if peak > self.count:
            raise ValueError("Error: Parameter peak can not be greater than count ")
        if peak < 1:
            raise ValueError("Peak can not be less than 0")
        else:
            prob = [0] * (peak - 1) + [1] + [0] * (self.count - peak)
            return(prob)
            
    def random_skew(self, smooth = "False", peak = 0):
        
        """
        Provides a skewed probability distribution. 
        Peak - Set the place with the highest prob.
            If peak is not specified a random place 
            will be determined at the right hand side
            of the distrubition.
        Smooth - False/ F : Probabilities changed lineary
                 True/ T  : Probabilities are smoothened
        """
    
        if peak > self.count:
            raise ValueError("Error: Parameter peak can not be greater than count ")
        if peak < 0:
            raise ValueError("Peak can not be less than 0")
        elif smooth not in ["False","F","True","T"]:
            raise ValueError("Error: Parameter smooth can only take values 'False','F','True','T' ")
        else:
            if peak == 0:
                peak = random.randint(int(self.count / 2), self.count - 2)     
            a = [x for x in range(1, self.count + 1) if  x <= peak]
            a.extend([((self.count - x + 1)/(self.count - peak + 1)) * peak  for x in range(1, self.count + 1) if x > peak])
            
            if smooth == "True" or smooth == "T":
                 a = [math.log(x) * x * x for x in a]
            prob = [x / sum(a) for x in a]
            if prob != 1:
                prob[prob.index(max(prob))] = max(prob)- sum(prob) + 1
            return(prob)
    
    def decreasing(self):
        """
        Provides lineary decreasing probability distribution. 
        """
        prob = [int(x) / (sum(range(1, self.count + 1))) for x in sorted(range(1, self.count + 1), reverse = True)]
        return(prob)
    
    def increasing(self):
        """
        Provides lineary increasing probability distribution. 
        """
        prob = [int(x) / (sum(range(1, self.count + 1))) for x in range(1, self.count + 1)]
        return(prob)
       
    def binom(self, n, po):
        """
        Provides probability distribution based on binominal distribution
        n  - number of trials, n an integer >= 0
        po - probability of success, po is in the interval [0,1]
        """
        prob = stats.binom.pmf(k = range(0, self.count), n = n, p = po)
        prob = [round(x / sum(prob), 15) for x in prob]
        if sum(prob) != 1:
            prob[prob.index(max(prob))] = max(prob)- sum(prob) +1
        return(prob)
    
    def skewnorm(self, a):
        """
        Provides probability distribution based on skewed standard normal
        a - skewness parameter, (When a = 0 it is identical to a normal distribution)
        """
        x = np.linspace(stats.skewnorm.ppf(0.0001, a = a),
                stats.skewnorm.ppf(0.9999, a = a), self.count)
        prob = stats.skewnorm.pdf(x, a = a)
        
        prob = [round(x / sum(prob), 15) for x in prob]
        if sum(prob) != 1:
            prob[prob.index(max(prob))] = max(prob)- sum(prob) + 1
        return(prob)
    
    def t(self, df):
        """
        Provides probability distribution based on t distribution
        df - degrees of freedom paramete
        """
        x = np.linspace(stats.t.ppf(0.0001, df = df),
                stats.t.ppf(0.9999, df = df), self.count)
        prob = stats.t.pdf(x, df = df)
        prob = [round(x / sum(prob), 15) for x in prob]
        if sum(prob) != 1:
            prob[prob.index(max(prob))] = max(prob)- sum(prob) + 1
        return(prob)    
    
    def alpha(self, a):
        """
        Provides probability distribution based on alpha distribution
        a - shape parameter
        """
        x = np.linspace(stats.alpha.ppf(0.0001, a = a),
                stats.alpha.ppf(0.9999, a = a), self.count)
        prob = stats.alpha.pdf(x, a = a)

        prob = [round(x / sum(prob), 15) for x in prob]
        if sum(prob) != 1:
            prob[prob.index(max(prob))] = max(prob)- sum(prob) + 1
        return(prob)    
    
    def gamma(self, a):
        """
        Provides probability distribution based on gamma distribution
        a - shape parameter
        """
        x = np.linspace(stats.gamma.ppf(0.0001, a = a),
                stats.gamma.ppf(0.9999, a = a), self.count)
        prob = stats.gamma.pdf(x, a = a)
        
        prob = [round(x / sum(prob),15) for x in prob]
        if sum(prob) != 1:
            prob[prob.index(max(prob))] = max(prob)- sum(prob) + 1
        return(prob) 
    
    def beta(self, a, b):
        """
        Provides probability distribution based on beta distribution
        a,b - shape parameter
        """
        x = np.linspace(stats.beta.ppf(0.0001, a = a, b = b),
                stats.beta.ppf(0.9999, a = a, b = b), self.count)
        prob = stats.beta.pdf(x, a = a, b = b)        
        
        prob = stats.beta.pdf(a = a, b = b, x = range(0, self.count))
        prob = [round(x / sum(prob), 15) for x in prob]
        if sum(prob) != 1:
            prob[prob.index(max(prob))] = max(prob) - sum(prob) + 1
        return(prob) 

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import warnings
import spicy
from scipy import stats