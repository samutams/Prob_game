
# coding: utf-8

class Risky_bomb_opt(object):
    
    """
    This is a docstring for the Risky_bomb_opt class.
    The class contains couple of optimization solutions for the probs game.
    count - How many number can be selected during the game
    probs - Type of the probability density function
    change_bomb - Change the place of the bomb by rounds
    n_play - Number of rounds
    
    """
    
    def __init__(self,
                 count, 
                 probs,  
                 change_bomb  = "F",
                 n_play = 10,
                 **kwargs):
        
        self.count = count
        self.base = Risky_bomb(count = count,
                               probs = probs,
                               a = kwargs.get('a'),
                               peak = kwargs.get('peak'),
                               smooth = kwargs.get('smooth'),
                               n = kwargs.get('n'),
                               po = kwargs.get('po'),
                               df = kwargs.get('df'))
        
        self.p = self.base.p
        self.cp = np.cumsum(self.p)
        self.change_bomb = change_bomb
        if n_play < 1:
            raise ValueError("Parameter n_play must be at least 1")
        self.n_play = n_play
        self.w_input = None
        self.w_output = None
        
    def plus_min_x(self, plus_min = 5):
        """
        Adds or subtracts a value depending if the payment was positive or negative in the previous round.
        plus_min - Value to be subtracted or added
        """   
        guess_in = [np.random.randint(0, self.count)]
        p = [self.base.payment(guess = guess_in[0], change_bomb = self.change_bomb)]
        for i in range(0, self.n_play):  
            if p[i] > 0: 
                guess_in.append(guess_in[i] + plus_min)
                p.append(self.base.payment(guess = guess_in[i + 1], change_bomb = self.change_bomb))
            if p[i] == 0:
                guess_in.append(guess_in[i] - plus_min)
                p.append(self.base.payment(guess = guess_in[i + 1], change_bomb = self.change_bomb))
        out_ = pd.DataFrame({"payment":p,"guesses":guess_in})
        return(out_)
    
    def fix_bomb_opt(self):
        """
        Quickly finding the optimum for cases when the bomb is fixed at one position over the games. 
        """   
        guess_in = [np.random.randint(1, self.count)]
        p = [self.base.payment(guess = guess_in[0], change_bomb = self.change_bomb)]
        min_ = 1
        max_ = self.count
        for i in range(0, self.n_play):
            p_h = np.array(p)
            g_h = np.array(guess_in)
            try: 
                min_ = max(g_h[np.where(p_h > 0)])
            except ValueError:
                min_ = min_
            try:
                max_ = min(g_h[np.where(p_h == 0)])
            except ValueError:
                max_ = max_
            guess_in.append(np.random.randint(min_, max_))
            p.append(self.base.payment(guess = guess_in[i + 1],change_bomb = self.change_bomb))
        out_ = pd.DataFrame({"payment":p, "guesses":guess_in})
        return(out_)
    
    def mean_opt(self):
        """
        Quickly finding the optimum for cases when the bomb is fixed at one position over the games. 
        """   
        guess_in = [np.random.randint(1, self.count)]
        p = [self.base.payment(guess = guess_in[0] , change_bomb = self.change_bomb)]
        opt_ = 25
        learn_ = int(self.n_play / 3)
        for j in range(1,learn_):
            guess_in.append(np.random.randint(1, self.count))
            p.append(self.base.payment(guess = guess_in[j], change_bomb = self.change_bomb))

        for i in range(learn_, self.n_play - learn_):
            p_h = np.array(p)
            g_h = np.array(guess_in)
            try: 
                opt_ = int(np.mean(list(p_h[np.where(p_h > 0)])))
            except ValueError:
                opt_ = opt_
            guess_in.append(opt_)
            p.append(self.base.payment(guess = guess_in[i], change_bomb = self.change_bomb))
        out_ = pd.DataFrame({"payment":p, "guesses":guess_in})
        return(out_)
    
    def nn_opt(self, 
               n_input = 1, 
               n_hidden = 5, 
               n_output = 1, 
               add_input = None, 
               w_input = None, 
               w_output = None):
        """
        Uses nn with random initial weights to play the game over the rounds.
        n_input - Number of input nodes
        n_hidden - Number of hidden nodes
        n_output - Number of output nodes
        add_input - Statistics that should be considered during the nn training
        w_input - Update of the input weights
        w_output - Update of the output weights
        """      
        # Initiate nn
        if add_input != None:
            nn = Nn_obj(count = self.count,
                        n_input = n_input, 
                        n_hidden = n_hidden, 
                        n_output = n_output, 
                        add_input = [0] * len(add_input))
        else:
            nn = Nn_obj(count = self.count,
                        n_input = n_input, 
                        n_hidden = n_hidden, 
                        n_output = n_output)
        
        if w_input != None and w_output != None:
            nn.weight_update(w_input = np.asarray(w_input), w_output = np.asarray(w_output))
        elif w_input == None and w_output == None:
            pass
        else: 
            raise ValueError("w_input and w_output must be specified together ")
            
        # Initiate values 
        guess_in = [nn.guess]
        p = [self.base.payment(guess = guess_in[0], change_bomb = self.change_bomb)]
        bomb = [self.base.bomb]
        mean_guess = [np.mean(guess_in)]
        std_guess = [np.std(guess_in)]
        mean_p = [np.mean(p)]
        std_p = [np.std(p)]
        if len(p) < 3:
            c = 0
        else: 
            c = len(p) - 3
            
        last_3_p_mean = [np.mean(p[c:])]
        
                             
        for i in range(1, self.n_play):  

            nn.nn()
            guess_in.append(nn.guess)
            p.append(self.base.payment(guess = guess_in[i], change_bomb = self.change_bomb))
            bomb.append(self.base.bomb)
            mean_guess.append(np.mean(guess_in))
            std_guess.append(np.std(guess_in))
            mean_p.append(np.mean(p))
            std_p.append(np.std(p))
            if len(p) < 3:
                c = 0
            else: 
                c = len(p) - 3
            last_3_p_mean.append(np.mean(p[c:]))
            
            attr = {'mean_guess': mean_guess[i], 
                    'std_guess': std_guess[i], 
                    'mean_p': mean_p[i], 
                    'std_p': std_p[i],
                    'last_3_p_mean': last_3_p_mean[i], 
                    'last_p': p[i]}
            
            if add_input != None:
                nn.add_input_update(add_input_ = [attr[x] for x in add_input])
                                 
        out_ = pd.DataFrame({"payment": p,
                             "guesses": guess_in, 
                             "bomb":bomb,
                             "mean_guess": mean_guess, 
                             "std_guess": std_guess,
                             "mean_p": mean_p, 
                             "std_p": std_p,
                             "last_3_p_mean": last_3_p_mean})
        
        self.w_input = nn.w_input
        self.w_output = nn.w_output
        return(out_)
    
    
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
from Prob_game.nn_obj import *