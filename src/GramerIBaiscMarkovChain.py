
import logging
import os
import json
import time
import numpy as np
from BaseGramerIt import BaseGramerIt

class GramerIBaiscMarkovChain(BaseGramerIt):
    def __init__ (self, model_path='weights/'):
        super().__init__(model_path)
    
    def load_model(self):
        logging.debug('Loading the model from {}'.format(self.model_folder_path))
        time_start = time.time()
        self.bigram_probab = json.load(open(os.path.join(self.model_folder_path, 'bigramprobab2.json')))
        self.V = len(self.bigram_probab.keys())
        time_end = time.time()
        logging.debug('Loaded the model from {}, time taken {}'.format(self.model_folder_path, time_end - time_start))
  
    def predict_a_b(self, a, b):
        if a not in self.bigram_probab.keys() or b not in self.bigram_probab[a].keys():
            return np.log(1 / self.V)   # Add-one smoothing
        else:
            return np.log(self.bigram_probab[a][b])
