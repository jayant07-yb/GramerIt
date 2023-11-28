import sys
import argparse
import logging
import re
import os
import json
import time
import numpy as np
from BaseGramerIt import BaseGramerIt
from vectorizer import GloveVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class GramerItDeepLearning(BaseGramerIt):
    def __init__ (self, model_path='weights/'):
        super().__init__(model_path)
        self.vectorizer = GloveVectorizer()
        self.model = self.get_model()
        self.load_model()

    def get_model(self, input_shape=100):
        model = Sequential()
        model.add(Dense(50, input_dim=input_shape, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        return model

    def load_model(self):
        logging.debug('Loading the model from {}'.format(self.model_folder_path))
        time_start = time.time()
        
        self.model = self.get_model()
        self.model.load_weights(os.path.join(self.model_folder_path, 'bigram_model_2.h5'))

        time_end = time.time()
        logging.debug('Loaded the model from {}, time taken {}'.format(self.model_folder_path, time_end - time_start))
  
    def predict_a_b(self, a, b):
        a = self.vectorizer.transform_word(a)
        b = self.vectorizer.transform_word(b)
        logging.debug('a: {}, b: {}'.format(a, b))
        combined_embedding = np.concatenate([a, b])

        return self.model.predict(combined_embedding.reshape(1, -1), verbose=0)
    