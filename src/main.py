import argparse
import logging
import sys

from GramerIBaiscMarkovChain import GramerIBaiscMarkovChain
from GramerItDeepLearning import GramerItDeepLearning
from GramerItLinearRegression import GramerItLinearRegression

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
            predictor = GramerItLinearRegression(args.model_path)
        else:
            predictor = GramerItLinearRegression()
    elif args.model == 'deep_learning':
        predictor = GramerItDeepLearning()
    
    elif args.model == 'basic_markov_chain':
        if args.model_path:
            predictor = GramerIBaiscMarkovChain(args.model_path)
        else:
            predictor = GramerIBaiscMarkovChain()
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
