"""
This file is used to test the performance of various model.
"""
import TestData as testD
import logging
import time
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from GramerIBaiscMarkovChain import GramerIBaiscMarkovChain
from GramerItDeepLearning import GramerItDeepLearning
from GramerItLinearRegression import GramerItLinearRegression

output_dict = {
    'basic_markov_chain': {
        'ai_generated': {
            'score' : [],
            'time_taken': [],
            'length': [],
        },
        'random': {
            'score' : [],
            'time_taken': [],
            'length': [],
        },
        'human': {
            'score' : [],
            'time_taken': [],
            'length': [],
        },
    },
    'linear_regression': {
        'ai_generated': {
            'score' : [],
            'time_taken': [],
            'length': [],
        },
        'random': {
            'score' : [],
            'time_taken': [],
            'length': [],
        },
        'human': {
            'score' : [],
            'time_taken': [],
            'length': [],
        },
    },
    'deep_learning': {
        'ai_generated': {
            'score' : [],
            'time_taken': [],
            'length': [],
        },
        'random': {
            'score' : [],
            'time_taken': [],
            'length': [],
        },
        'human': {
            'score' : [],
            'time_taken': [],
            'length': [],
        },
    },
}


# First load the model before starting the test
basic_markov_chain = GramerIBaiscMarkovChain()
linear_regression = GramerItLinearRegression()
deep_learning = GramerItDeepLearning()

def append_perf(sentence_type, sentences):
    for sentence in sentences:
        if len(basic_markov_chain.process_sentence(sentence)) < 2:
            continue

        logging.info('Testing the sentence: {}'.format(sentence))
        time_start = time.time()
        output_dict['basic_markov_chain'][sentence_type]['score'].append(basic_markov_chain.predict(sentence))
        output_dict['basic_markov_chain'][sentence_type]['time_taken'].append(time.time() - time_start)
        output_dict['basic_markov_chain'][sentence_type]['length'].append(len(sentence))
        
        time_start = time.time()    
        output_dict['linear_regression'][sentence_type]['score'].append(linear_regression.predict(sentence))
        output_dict['linear_regression'][sentence_type]['time_taken'].append(time.time() - time_start)
        output_dict['linear_regression'][sentence_type]['length'].append(len(sentence))

        time_start = time.time()
        output_dict['deep_learning'][sentence_type]['score'].append(deep_learning.predict(sentence))
        output_dict['deep_learning'][sentence_type]['time_taken'].append(time.time() - time_start)
        output_dict['deep_learning'][sentence_type]['length'].append(len(sentence))

pairs = {
    'ai_generated': 'AI Generated text',
    'random': 'Random text',
    'human': 'Human written text',
}

color = {
    'ai_generated': 'ro',
    'random': 'bo',
    'human': 'go',
}
def plot_score(axes, model_type):


    for key in pairs.keys():
        # Reshape data if needed
        length = output_dict[model_type][key]['length']
        score = output_dict[model_type][key]['score']
        length = np.reshape(length, (len(length),))  # Reshape to 1D array if needed
        score = np.reshape(score, (len(score),))  # Reshape to 1D array if needed

        # Scatter plot for 'ai_generated'
        axes.plot(length, score, color[key], label=pairs[key])


    axes.set_xlabel('Length of the sentence')
    axes.set_ylabel('Score of the sentence')
    axes.set_title('Scatter Plot of Scores vs Lengths')
    axes.legend()

def create_plot():
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))

    # Plot for 'basic_markov_chain'
    plot_score(axes[0][0], 'basic_markov_chain')
    axes[0][0].set_title('Basic Probability')

    # Plot for 'linear_regression'
    plot_score(axes[0][1], 'linear_regression')
    axes[0][1].set_title('Linear Regression')

    # Plot for 'deep_learning'
    plot_score(axes[0][2], 'deep_learning')
    axes[0][2].set_title('Deep Learning')

    # Plot the time taken for each model
    axes[1][1].plot(output_dict['basic_markov_chain']['ai_generated']['length'], output_dict['basic_markov_chain']['ai_generated']['time_taken'], 'ro', label='Basic Probability')
    axes[1][1].plot(output_dict['linear_regression']['ai_generated']['length'], output_dict['linear_regression']['ai_generated']['time_taken'], 'bo', label='Linear Regression')
    axes[1][1].plot(output_dict['deep_learning']['ai_generated']['length'], output_dict['deep_learning']['ai_generated']['time_taken'], 'go', label='Deep Learning')
    axes[1][1].set_xlabel('Length of the sentence')
    axes[1][1].set_ylabel('Time taken to predict')
    axes[1][1].set_title('Time taken to predict vs Lengths')
    axes[1][1].legend()


    # Hide the axes in the second row
    axes[1][0].axis('off')
    axes[1][2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    append_perf('ai_generated', testD.get_ai_generated_sentences())
    append_perf('random', testD.get_random_sentences())
    append_perf('human', testD.get_human_sentences())
 
    logging.info('Performance of the models', output_dict)
    print(output_dict)

    
    # with open('performance.json', 'w') as json_file:
    #     json.dump(output_dict, json_file)

    create_plot()


    