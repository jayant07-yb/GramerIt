import sys
import argparse
import logging
import re
import os
import json
import time
import numpy as np
from vectorizer import GloveVectorizer

class BaseGramerIt():
    def __init__ (self, model_path='weights/'):
        self.model_folder_path = model_path
        self.load_model()
        self.report = []

    def load_model(self):
        pass

    def predict(self, text):
        filtered_words = self.process_sentence(text)
        if len(filtered_words) < 2:
            logging.error('Text is too short')
            return 0

        probab = self.sentence_prob(filtered_words)


        return self.sentence_prob(filtered_words)
        
    def predict_batch(self, text):
        pass
    
    def process_sentence(self, sentence):
        # Remove special characters, make lowercase
        sentence = sentence.expandtabs()
        words = [re.sub(r'[^a-zA-Z]', '', word).lower() for word in sentence.split()]
        

        # Remove numbers, blank words, and words with a single character
        filtered_words = [word for word in words if word and not word.isdigit() and len(word) > 1]
            
        return filtered_words

    def sentence_prob(self, sentence):
        """
        Returns the probability of the sentence
            Use the log probabilities normalized by the lenght of the sentence
        """
        if len(sentence) == 0:
            return 0
        log_prob = 0
        for i in range(len(sentence)):
            if i == len(sentence) - 1:
                continue # Do nothing p(a|b,a) 
            else:
                log_prob +=  self.predict_a_b(sentence[i], sentence[i+1])
        return log_prob / (len(sentence) - 1)

    def predict_a_b(self, a, b):
        """
        Returns the probability of b given a
        Where b is the next word and a is the previous word
        """
        pass


