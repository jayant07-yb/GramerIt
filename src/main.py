import sys
import argparse
import logging
import re
import os
import json
import time
import numpy as np
from vectorizer import GloveVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class Gramerly():
    def __init__ (self, model_path='weights/'):
        self.model_folder_path = model_path
        self.load_model()

    def load_model(self):
        pass

    def predict(self, text):
        filtered_words = self.process_sentence(text)
        if len(filtered_words) < 2:
            logging.error('Text is too short')
            sys.exit(1)

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

class GramerlyLinearRegression(Gramerly):
    def __init__ (self, model_path='weights/'):
        super().__init__(model_path)
        self.model = None

    def load_model(self):
        self.bigram_probab = json.load(open(os.path.join(self.model_folder_path, 'bigramprobab2.json')))

class GramerlyDeepLearning(Gramerly):
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

        return self.model.predict(combined_embedding.reshape(1, -1))
    
class GramerltBaiscMarkovChain(Gramerly):
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

if __name__ == '__main__':
    """"
    This function is used to predict the quality of a given text.
    For that purpose basic probability and machine learning is used.

    Idea here is to use bigram model to predict the quality of a given text.
    Wikipedia data is used to train the model.

    This prototype can be used to test various bigram models.
    
    """
    parser = argparse.ArgumentParser(description='Predict the quality of a given text.')
    parser.add_argument('--text', help='Text to be predicted')
    parser.add_argument('--model', help='Model to be used for prediction')
    parser.add_argument('--model_path', help='Path to the model')
    parser.add_argument('--log', help='Log level', default='INFO')
    # parser.add_argument('--help', help='Help', action='store_true')
    args = parser.parse_args()

    # Set the log level
    logging.basicConfig(level=args.log)

    # Check if the model is provided
    if not args.model:
        logging.error('Model is not provided')
        sys.exit(1)

    predictor = None
    if args.model == 'linear_regression':
        if args.model_path:
            predictor = GramerlyLinearRegression(args.model_path)
        else:
            predictor = GramerlyLinearRegression()
    elif args.model == 'deep_learning':
        predictor = GramerlyDeepLearning()
    
    elif args.model == 'basic_markov_chain':
        if args.model_path:
            predictor = GramerltBaiscMarkovChain(args.model_path)
        else:
            predictor = GramerltBaiscMarkovChain()
    else:
        logging.error('Invalid model provided')
        logging.error('Supported models are: linear_regression, deep_learning, basic_markov_chain')
        sys.exit(1)

    # Check the text
    if not args.text:
        logging.error('Text is not provided')
        sys.exit(1)
    
    if len(args.text) < 2:
        logging.error('Text is too short')
        sys.exit(1)
    
    logging.info("Quality of the text is {}".format(predictor.predict(args.text)))
