
# coding: utf-8

class Nn_evolution(object):
    """
    This is a docstring for the Nn_evolution class.
    The class finds an optimal nn algorithm for a selected prob game by evolutionary algorithm.
    count - How many number can be selected during the game
    probs - Type of the probability density function
    change_bomb - Change the place of the bomb by rounds
    
    n_play - Number of rounds
    n_pop - Size of the population in a generation
    elitism - Percent of best nn algorithms that will be evaluated in the next round
    n_gen - Number of generations
    loser_percent - Percent of worse performance that will be evaluated in the next round
    
    n_input - Number of input nodes
    n_hidden - Number of hidden nodes
    n_output - Number of output nodes
    add_input - Statistics that should be considered during the nn training
    """
    
    def __init__(self,
                 count,
                 probs,
                 change_bomb  = "T",
                 n_play = 10,
                 n_pop = 10, 
                 elitism = 0.2, 
                 n_gen = 5, 
                 loser_percent = 0.2,
                 # NN parameters
                 n_input = 5, 
                 n_hidden = 5, 
                 n_output = 1, 
                 add_input = ["mean_guess", "mean_p", "last_3_p_mean", "last_3_p_mean"],
                 **kwargs
                ):
        
        self.count = count
        self.probs = probs
        self.change_bomb = change_bomb
        self.n_play = n_play
        self.n_pop = n_pop
        self.elitism = elitism
        self.n_gen = n_gen
        self.loser_percent = loser_percent
        # NN
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.add_input = add_input
        
        self.evolution_df = pd.DataFrame()
        self.evolution_results = pd.DataFrame()
        
        self.best_results_all = pd.DataFrame()
        
        # Weights
        self.best_w_input = None
        self.best_w_output = None
        
    # Evolve nn functions
    def evolve(self, **kwargs):
        """
        Runs the evolutionary nn algorithm.
        """        
        # Initiate the output table
        df = pd.DataFrame(columns = ["ID", "w_input", "w_output", "score", "Sum_score","Mean_guess"])
        
        for j in tqdm(range(0, self.n_gen)):

            for i in range(0, self.n_pop):
                opt_inp = Risky_bomb_opt(count = self.count, 
                                         probs = self.probs, 
                                         change_bomb = self.change_bomb,
                                         a = kwargs.get('a'),
                                         peak = kwargs.get('peak'),
                                         smooth = kwargs.get('smooth'),
                                         n = kwargs.get('n'),
                                         po = kwargs.get('po'),
                                         df = kwargs.get('df'))
                if j == 0:
                    out_opt = opt_inp.nn_opt(n_input = self.n_input, 
                                             n_hidden = self.n_hidden, 
                                             n_output = self.n_output, 
                                             add_input = self.add_input )
                else:
                     out_opt = opt_inp.nn_opt(n_input = self.n_input, 
                                             n_hidden = self.n_hidden, 
                                             n_output = self.n_output, 
                                             add_input = self.add_input,
                                             w_input = list(df_new["w_input"][i]), 
                                             w_output = list(df_new["w_output"][i]))               


                df = df.append({'Generation': j,
                                'ID': i, 
                                "w_input": opt_inp.w_input, 
                                "w_output": opt_inp.w_output, 
                                "score": int(sum(out_opt["payment"]) / len(out_opt["payment"])),
                                "Sum_score": int(sum(out_opt["payment"])),
                                "Mean_guess": int(sum(out_opt["guesses"]) / len(out_opt["guesses"]))
                                }, ignore_index = True)

            # Elitism
            elitism_n = int(np.floor(self.elitism * self.n_pop))
            loser_n = int(np.floor(self.loser_percent * self.n_pop))
            df_c_gen = df[df['Generation'] == j].reset_index()
            df_c_gen["score"] = df_c_gen.score.astype(float)
            elits = df_c_gen.nlargest(elitism_n, "score")
            elits = elits[["w_input", "w_output"]]
            losers = df_c_gen.nsmallest(loser_n, "score")
            losers = losers[["w_input", "w_output"]]



            df_parents = pd.DataFrame(columns = ["w_input", "w_output"])
            df_parents = df_parents.append(elits, ignore_index = True)
            df_parents = df_parents.append(losers, ignore_index = True)


            # Breed
            n_parents = df_parents.shape[0]
            n_children = self.n_pop - n_parents
            df_children = pd.DataFrame(columns = ["w_input", "w_output"])
            for i in range(0, n_children):
                # w_input
                p1,p2 = random.randint(0, n_parents-1), random.randint(0, n_parents-1)
                w_1 = random.uniform(0, 1)
                w_2 = 1 - w_1
                w_input_new = df_parents["w_input"][p1] * w_1 + df_parents["w_input"][p2] * w_2
                w_output_new = df_parents["w_output"][p1] * w_1 + df_parents["w_output"][p2] * w_2
                # Caps
                w_input_new[w_input_new > 1] = 1
                w_input_new[w_input_new < -1] = -1
                w_output_new[w_output_new > 1] = 1
                w_output_new[w_output_new < -1] = -1 

                df_children = df_children.append({
                                "w_input": w_input_new, 
                                "w_output": w_output_new}, ignore_index=True)

            # Mutate 
            for i in range(0, n_children - 1):
                # w_input
                p1 = random.randint(0, n_children - 1)
                m_f = random.uniform(-0.25, 0.25)
                w_input_new = df_children["w_input"][p1] + m_f
                w_output_new = df_children["w_output"][p1] + m_f
                # Caps
                w_input_new[w_input_new > 1] = 1
                w_input_new[w_input_new < -1] = -1
                w_output_new[w_output_new > 1] = 1
                w_output_new[w_output_new < -1] = -1 

                df_children["w_input"][p1] = w_input_new
                df_children["w_output"][p1] = w_output_new

            df_new =  pd.DataFrame(columns = ["w_input", "w_output"])
            df_new = df_new.append(df_parents, ignore_index = True)
            df_new = df_new.append(df_children, ignore_index = True)
        
        id_help = []
        for i in range(0, len(df["ID"])):
            id_help.append(str(sum(np.dot(sum(df.w_output[i]), df.w_input[i]))))
        df['ID2'] = pd.factorize(id_help)[0]
        
        self.evolution_df = df
    
    def summary_results(self):
        """
        Creates a table that shows the improvement of the nn by generations.
        """    
        result = pd.DataFrame()
        result['sum'] = self.evolution_df.groupby('Generation')['score'].sum()
        result['avg'] = self.evolution_df.groupby('Generation')['score'].sum() / self.evolution_df.groupby('Generation')['score'].count()
        result['max'] = self.evolution_df.groupby('Generation')['score'].max()
        result['min'] = self.evolution_df.groupby('Generation')['score'].min()
        self.evolution_results = result
        return(result)
    
    def plot_improvement(self, plot_type ="avg"):     
        """
        Creates a plot that shows the improvement of the nn by generations.
        """    
        # Plotting
        plt.plot(range(1,len(self.evolution_results['sum']) + 1), self.evolution_results[plot_type], color = '#001871')
        plt.xlabel('Generation', fontsize = 15)
        plt.ylabel('Score', fontsize = 15)
        plt.title('Improvement in the model by nn generations' , fontsize = 20)
        plt.rcParams["figure.figsize"] = [10,7]

        plt.show() 
    
    def select_best_nn(self, 
                       gen_cons = 'all', 
                       variable = "Avg_avg_scores", 
                       min_gen = 5):
        """
        Select the best nn weights that could provide the best results. 
        gen_cons - Set the generation, from which the best model should be selected. (all - all generations are considered)
        variable - Controls that which variable should be assessed to select the best model.
        min_gen - The minimum number of generation where the best model should be present. 
                    (More survived generations the better long performance)
        """  
        # All stats
        result = pd.DataFrame()
        result["Sum_avg_scores"] = self.evolution_df.groupby('ID2')['score'].sum()
        result["Sum_sum_scores"] = self.evolution_df.groupby('ID2')['Sum_score'].sum()
        result["Sum_Mean_guess"] = self.evolution_df.groupby('ID2')['Mean_guess'].sum()
        result["Count_Gen"] = self.evolution_df.groupby('ID2')['score'].count()
        #
        result["Avg_avg_scores"] = result["Sum_avg_scores"] / result["Count_Gen"]
        result["Avg_sum_scores"] = result["Sum_sum_scores"] / result["Count_Gen"]
        result["Avg_Mean_guess"] = result["Sum_Mean_guess"] / result["Count_Gen"]
        result["Last_generation"] = self.evolution_df.groupby('ID2')['Generation'].max()
        
        un_df =  self.evolution_df.groupby('ID2').first()
        un_df = un_df[['w_input','w_output']]
        result = pd.merge(result, un_df, left_on = 'ID2', right_on = 'ID2').reset_index()
       
        # Filter and sort
        result = result[result["Count_Gen"] >= min_gen]
        result = result.sort_values(by = variable, ascending = False).reset_index()
        
        self.best_results_all = result
        
        if gen_cons == 'all':
            pass
        else: 
            result = result[result["Last_generation"] == self.n_gen - 1]
            
        self.best_w_input = result["w_input"][0]
        self.best_w_output = result["w_output"][0]
           
        return({'w_input': self.best_w_input, 'w_output': self.best_w_output})
        

        
from Prob_game.risky_bomb_opt import *
from Prob_game.risky_bomb_stats import *
from tqdm import tqdm
