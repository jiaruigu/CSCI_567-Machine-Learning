from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=1000, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = nb_features
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        for i in range(self.max_iteration):
            converged = True
            for n in range(len(features)):
                predict_label = 1 if (np.matmul(self.w, np.transpose([features[n]])) >= 0)[0] else -1
                if labels[n] != predict_label:
                    self.w = (np.array(self.w) + labels[n]/(np.inner(features[n], features[n])**0.5)* np.array(features[n])).tolist()
                    converged = False
            if converged == True:
                return True
        return False

    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[float]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################
        values = np.matmul(self.w,np.transpose(features))
        for i in range(len(values)):
            values[i] = 1.0 if values[i] >= 0 else -1
        return values

    def get_weights(self) -> List[float]:
        return self.w
    