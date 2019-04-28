
# coding: utf-8

class Sample_generation(object):
    
    """
    This is a docstring for the Sample_generation class.
    The class generate training sample for a given probability density function.
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
                               n =  kwargs.get('n'),
                               po =  kwargs.get('po'),
                               df =  kwargs.get('df'))
        
        self.p = self.base.p
        self.cp = np.cumsum(self.p)
        self.change_bomb = change_bomb
        if n_play < 1:
            raise ValueError("Parameter n_play must be at least 1")
        self.n_play = n_play
        self.w_input = None
        self.w_output = None
        
        self.gen_sample_df = pd.DataFrame()
    
    def gen_sample(self, flag = 1, n_sample = 5):
        """
        Generate random sample for a given prob game with definied prob desnsity function. 
        flag - assigned indentifier in the output table
        n_sample - number of games to be generated
        """
        df_out = pd.DataFrame(columns = ["Sample_ID",
                                         "seq", 
                                         "flag",
                                         "payment",
                                         "guesses",
                                         "bomb",
                                         "mean_guess",
                                         "std_guess",
                                         "mean_p",
                                         "std_p",
                                         "last_3_p_mean"])
        for j in range(0, n_sample - 1):
            
             # Initiate values
            seq_ = []
            flag_ = []
            guess_in = []
            p = []
            bomb = []
            mean_guess = []
            std_guess = []
            mean_p = []
            std_p = []
            last_3_p_mean= []

            for i in range(0, self.n_play):  
                seq_.append(i)
                flag_.append(flag)
                guess_in.append(np.random.randint(0,self.count))
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
                
            out_ = pd.DataFrame({"Sample_ID": j,
                                 "seq": seq_,
                                 "flag": flag_,
                                 "payment": p,
                                 "guesses": guess_in, 
                                 "bomb":bomb,
                                 "mean_guess": mean_guess, 
                                 "std_guess": std_guess,
                                 "mean_p": mean_p, 
                                 "std_p": std_p,
                                 "last_3_p_mean": last_3_p_mean})
            
            df_out = df_out.append(out_ , ignore_index = True)
        
        self.gen_sample_df = df_out
        return(df_out)


from Prob_game.risky_bomb import *