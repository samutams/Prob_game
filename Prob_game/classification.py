
# coding: utf-8

class Classification_type(object):
        
    """
    This is a docstring for the Classification_type class.
    The class creates a simple nn based on a generated prob sample, that enables to classify new observations 
    to the previously seen classes.
    dataset - Generated training sample.
    """
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.y = dataset['flag'].values
        self.X = dataset[['seq', 'payment', 'guesses', 'mean_guess', 'std_guess', 'mean_p', 'std_p', 'last_3_p_mean']]
        self.n_out_neurons = self.dataset['flag'].nunique()
        
        # Y edited
        self.dummy_y = None
        
        # X edited
        self.X_scaled = None
    
        # Classification NN
        self.classifier = Sequential()
        self.classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
        #self.classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
        self.classifier.add(Dense(units = self.n_out_neurons,  activation = 'softmax'))
        self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        # Prediction
        self.y_pred = None
        
        # Score by Seq
        self.score_seq_dt = None
    
    def encode_y(self, y):
        """
        Encodes y to new dummy variables, that enables to run the nn algorithm.
        y - y values to be encoded.
        """
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_y = encoder.transform(y)
        return(np_utils.to_categorical(encoded_y))
        
    def scaling_x(self, x):
        """
        Performs scalling of the x variables.
        x - x values to be scaled
        """       
        sc = StandardScaler()
        return(sc.fit_transform(x))
        
    def classification_nn(self):
        """
        Training of the model, adds predicted value to the class.
        """     
        # Scale and Encode if needed
        if self.dummy_y == None: 
            self.dummy_y = self.encode_y(self.y)
        if self.X_scaled == None:
            self.X_scaled = self.scaling_x(self.X.values)
        
        # Classification
        self.classifier.fit(self.X_scaled, self.dummy_y, batch_size = 100, epochs = 100, verbose=0)
        
        # Add predicted Value to dataset
        y = self.classifier.predict(self.X_scaled)
        self.dataset['y_pred'] = self.prob_to_pred(y)
        self.dataset.loc[(self.dataset['y_pred'] == self.dataset['flag']) , 'Score'] = 1 
        self.dataset.loc[(self.dataset['y_pred'] != self.dataset['flag']), 'Score'] = 0 
        
        print("Model is trained")
    
    def prob_to_pred(self, y_output):
        """
        Converts prediction matrix to actual prediction
        y_output - y values to be converted
        """   
        out_ = []
        for i in range(0, y_output.shape[0]):
            out_.append((y_output[i].argmax(axis = 0) + 1))
        return(out_)
    
    def confusion_matrix(self, 
                         y_pred_, 
                         y_actual, 
                         normalize = True):
        """
        Creates confusion matrix, for given y predicted and observed values. 
        y_pred_ - Predicted value
        y_actual - Observed value
        normalize - Outputs normalized values in the confusion matrix.
        """  
        c = confusion_matrix(y_actual, y_pred_)
        
        if normalize == True:
            c = c / c.astype(np.float).sum(axis=1)
        return(c)
    
    def overall_score(self, 
                      score_type = 'Training',
                      y_act = None, 
                      y_pred = None):
        """
        Calculates the score of the model. 
        score_type - Proviedes the score value based on the training sample
        
        y_act - Observed value
        normalize - Outputs normalized values in the confusion matrix.
        """         
        if score_type == 'Training':
            return(self.dataset["Score"].sum() / self.dataset["Score"].count())
        else:
            y_eval = (y_act == y_pred)
            return(y_eval[y_eval == True].shape[0] / y_eval.shape[0])
    
    def score_by_seq(self):
        """
        Calculate the accuracy score by number of rounds.
        (Increasing trend is expected - as more rounds is played more characteristic information is observed)
        """ 
        self.score_seq_dt = self.dataset.groupby("seq")["Score"].sum() / self.dataset.groupby("seq")["Score"].count()
        return(self.score_seq_dt)
    
    def plot_score_by_seq(self): 
        """
        Plots the accuracy score by number of rounds.
        (Increasing trend is expected - as more rounds is played more characteristic information is observed)
        """ 
        self.score_seq_dt = self.dataset.groupby("seq")["Score"].sum() / self.dataset.groupby("seq")["Score"].count()
        
        plt.plot(range(0,len(self.score_seq_dt)), self.score_seq_dt , color = '#001871')
        plt.xlabel('N_play', fontsize = 15)
        plt.ylabel('% Correctly classified', fontsize = 15)
        plt.title('% of correctly classified cases depending on n_play', fontsize = 20)
        plt.rcParams["figure.figsize"] = [10,7]
        plt.show() 
    
    def predict_value(self, X_input):
        """
        Provides the predicted y value based on unscalled X inputs
        X_input - Unscaled x values
        """
        return(self.classifier.predict(self.scaling_x(X_input)))

    

from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')