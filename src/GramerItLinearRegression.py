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


class GramerItLinearRegression(BaseGramerIt):
    def __init__ (self, model_path='weights/'):
        super().__init__(model_path)
        self.vectorizer = GloveVectorizer()
        self.load_model()
        
    def load_model(self):
        self.weights = np.load(os.path.join(self.model_folder_path, 'linear_regression_weights.txt.npy'))
        self.bias = np.load(os.path.join(self.model_folder_path, 'linear_regression_bias.txt.npy'))

    def predict_a_b(self, a, b):
        a = self.vectorizer.transform_word(a)
        b = self.vectorizer.transform_word(b)
        logging.debug('a: {}, b: {}'.format(a, b))
        X = np.concatenate([a, b])
        return X.dot(self.weights) + self.bias