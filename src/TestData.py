"""
This file contains only the test data.
"""
import numpy as np

def get_human_sentences():
    return """
    The quick brown fox jumps over the lazy dog. 
    From Wikipedia, the free encyclopedia
    Part of a series on
    Regression analysis
    Models
    Linear regressionSimple regressionPolynomial regressionGeneral linear model
    Generalized linear modelVector generalized linear modelDiscrete choiceBinomial regressionBinary regressionLogistic regressionMultinomial logistic regressionMixed logitProbitMultinomial probitOrdered logitOrdered probitPoisson
    Multilevel modelFixed effectsRandom effectsLinear mixed-effects modelNonlinear mixed-effects model
    Nonlinear regressionNonparametricSemiparametricRobustQuantileIsotonicPrincipal componentsLeast angleLocalSegmented
    Errors-in-variables
    Estimation
    Least squaresLinearNon-linear
    OrdinaryWeightedGeneralizedGeneralized estimating equation
    PartialTotalNon-negativeRidge regressionRegularized
    Least absolute deviationsIteratively reweightedBayesianBayesian multivariateLeast-squares spectral analysis
    Background
    Regression validationMean and predicted responseErrors and residualsGoodness of fitStudentized residualGauss–Markov theorem
    icon Mathematics portal
    vte
    In statistics, linear regression is a linear approach for modelling a predictive relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables), which are measured without error. The case of one explanatory variable is called simple linear regression; for more than one, the process is called multiple linear regression.[1] This term is distinct from multivariate linear regression, where multiple correlated dependent variables are predicted, rather than a single scalar variable.[2]If the explanatory variables are measured with error then errors in variables models are required, also known as measurement error models.

    In linear regression, the relationships are modeled using linear predictor functions whose unknown model parameters are estimated from the data. Such models are called linear models.[3] Most commonly, the conditional mean of the response given the values of the explanatory variables (or predictors) is assumed to be an affine function of those values; less commonly, the conditional median or some other quantile is used. Like all forms of regression analysis, linear regression focuses on the conditional probability distribution of the response given the values of the predictors, rather than on the joint probability distribution of all of these variables, which is the domain of multivariate analysis.

    Linear regression was the first type of regression analysis to be studied rigorously, and to be used extensively in practical applications.[4] This is because models which depend linearly on their unknown parameters are easier to fit than models which are non-linearly related to their parameters and because the statistical properties of the resulting estimators are easier to determine.

    Linear regression has many practical uses. Most applications fall into one of the following two broad categories:

    If the goal is error reduction in prediction or forecasting, linear regression can be used to fit a predictive model to an observed data set of values of the response and explanatory variables. After developing such a model, if additional values of the explanatory variables are collected without an accompanying response value, the fitted model can be used to make a prediction of the response.
    If the goal is to explain variation in the response variable that can be attributed to variation in the explanatory variables, linear regression analysis can be applied to quantify the strength of the relationship between the response and the explanatory variables, and in particular to determine whether some explanatory variables may have no linear relationship with the response at all, or to identify which subsets of explanatory variables may contain redundant information about the response.
    Linear regression models are often fitted using the least squares approach, but they may also be fitted in other ways, such as by minimizing the "lack of fit" in some other norm (as with least absolute deviations regression), or by minimizing a penalized version of the least squares cost function as in ridge regression (L2-norm penalty) and lasso (L1-norm penalty). Use of the Mean Squared Error(MSE) as the cost on a dataset that has many large outliers, can result in a model that fits the outliers more than the true data due to the higher importance assigned by MSE to large errors. So a cost functions that are robust to outliers should be used if the dataset has many large outliers. Conversely, the least squares approach can be used to fit models that are not linear models. Thus, although the terms "least squares" and "linear model" are closely linked, they are not synonymous.
    Content
    This article will cover:
    * Downloading and loading the pre-trained vectors
    * Finding similar vectors to a given vector
    * “Math with words”
    * Visualizing the vectors
    Further reading resources, including the original GloVe paper, are available at the end.
    Brief Introduction to GloVe
    Global Vectors for Word Representation, or GloVe, is an “unsupervised learning algorithm for obtaining vector representations for words.” Simply put, GloVe allows us to take a corpus of text, and intuitively transform each word in that corpus into a position in a high-dimensional space. This means that similar words will be placed together.
    If you would like a detailed explanation of how GloVe works, linked articles are available at the end.
    Downloading Pre-trained Vectors
    Head over to https://nlp.stanford.edu/projects/glove/.
    Then underneath “Download pre-trained word vectors,” you can choose any of the four options for different sizes or training datasets.
    I have chosen the Wikipedia 2014 + Gigaword 5 vectors. You can download those exact vectors at http://nlp.stanford.edu/data/glove.6B.zip (WARNING: THIS IS A 822 MB DOWNLOAD)
    I cannot guarantee that the methods used below will work with all of the other pre-trained vectors, as they have not been tested.
    Imports
    We’re going to need to use, Numpy, Scipy, Matplotlib, and Sklearn for this project.
    If you need to install any of these, you can run the following:
    pip install numpy
    pip install scipy
    pip install matplotlib
    pip install sklearn
    Depending on your version of Python, you may need to substitute pip for pip3.

    Now we can import the parts we need from these modules with:


    Loading the Vectors
    Before we load the vectors in code, we have to understand how the text file is formatted.
    Each line of the text file contains a word, followed by N numbers. The N numbers describe the vector of the word’s position. N may vary depending on which vectors you downloaded, for me, N is 50, since I am using glove.6B.50d.

    Here is an example line from the text file, shortened to the first three dimensions:

    business 0.023693 0.13316 0.023131 ...
    To load the pre-trained vectors, we must first create a dictionary that will hold the mappings between words, and the embedding vectors of those words.

    embeddings_dict = {}
    Assuming that your Python file is in the same directory as the GloVe vectors, we can now open the text file containing the embeddings with:

    with open("glove.6B.50d.txt", 'r', encoding="utf-8") as f:
    Note: you will need to replace glove.6B.50d.txt with the name of the text file you have chosen for the vectors.

    Once inside of the with statement, we need to loop through each line in the file, and split the line by every space, into each of its components.

    After splitting the line, we make the assumption the word does not have any spaces in it, and set it equal the first (or zeroth) element of the split line.

    Then we can take the rest of the line, and convert it into a Numpy array. This is the vector of the word’s position.

    Finally, we can update our dictionary with the new word and its corresponding vector.


    As a recap for our full code to load the vectors:


    Keep in mind, you may need to edit the method for separating the word from the vectors if your vector text file includes words with spaces in them.

    Finding Similar Vectors
    Another thing we can do with GloVe vectors is find the most similar words to a given word. We can do this with a fancy one-liner function as follows:


    This one’s complicated, so let’s break it down.
    sorted takes an iterable as input and sorts it using a key. In this case, the iterable that we are passing in is all possible words that we want to sort. We can get a list of such words by calling embeddings_dict.keys().

    Now, since by default Python would sort the list alphabetically, we must specify a key to sort the list the way we want it sorted.
    In our case, the key will be a lambda function that takes a word as input and returns the distance between that word’s embedding and the embedding we gave the function. We will be using euclidean distance to measure how far apart the two embeddings are.

    scipy has a function for measuring euclidean distance under its module spatial, which we imported earlier. So our final sorting key turns into:

    lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding)
    Now if we want to rank all words by closeness to a given word, let’s say “king,” we can use:

    find_closest_embeddings(embeddings_dict["king"])
    This, however, will print every word, so if we want to shorten it we can use a slice at the end, for the closest, let’s say five words.

    find_closest_embeddings(embeddings_dict["king"])[:5]
    Since the closest word to a given word will always be that word, we can offset our slice by one.

    find_closest_embeddings(embeddings_dict["king"])[1:6]
    Using my vectors, glove.6B.50d,

    print(find_closest_embeddings(embeddings_dict["king"])[1:6])
    prints: [‘prince’, ‘queen’, ‘uncle’, ‘ii’, ‘grandson’]

    The reason we take an embedding directly, instead of transforming a word into an embedding, is so that when we add and subtract embeddings, we can find the closest approximate words to an embedding, not just a word. We can do this, even if the embedding does not lie entirely on any word.

    Math with Words
    Now that we can turn any word into a vector, we can use any math operation usable on vectors, on words.

    For example, we can add and subtract two words together, just like numbers. i.e., twig-branch+hand ≈ finger


    The above code prints “fingernails” as its top result, which is certainly passable as logical.

    Visualizing the Vectors
    Nothing helps to find insights in data more than visualizing it.

    To visualize the vectors, we are first going to be using a method known as t-distributed stochastic neighbor embedding, also known as t-SNE. t-SNE will allow us to reduce the, in my case, 50 dimensions of the data, down to 2 dimensions. After we do that, it’s as simple as using a matplotlib scatter plot to plot it. If you would like to learn more about t-SNE, there are a few articles linked at the end.

    sklearn luckily has a t-SNE class that can make our work much more manageable. To instantiate it, we can use:

    tsne = TSNE(n_components=2, random_state=0)
    n_components specifies the number of dimensions to reduce the data into.
    random_state is a seed we can use to obtain consistent results.

    After initializing the t-SNE class, we need to get a list of every word, and the corresponding vector to that word.

    words =  list(embeddings_dict.keys())
    vectors = [embeddings_dict[word] for word in words]
    The first line takes all the keys of embeddings_dict and converts it to a list.

    The second line uses list comprehension to obtain the value in embeddings_dict that corresponds to each word we chose, and put that into list form.

    We can also manually specify words so that it will only plot individual words. i.e., words = [“sister”, “brother”, “man”, “woman”, “uncle”, “aunt”]

    After getting all the words we want to use and their corresponding vectors, we now need to fit the t-SNE class on our vectors.
    We can do this using:

    Y = tsne.fit_transform(vectors[:1000])
    If you would like, you can remove or expand the slice at the end of vectors, but be warned; this may require a powerful computer.

    After the t-SNE class finishes fitting to the vectors, we can use a matplotlib scatter plot to plot the data:

    plt.scatter(Y[:, 0], Y[:, 1])
    This alone isn’t very useful since it’s just a bunch of dots. To improve it we can annotate the graph by looping through each X Y point with a label and calling plt.annotate with those X Y points and with that label. The other inputs to the function are for important formatting. Annotation in Matplotlib


    Finally, we can show the plot with,
    plt.show()
    A bit crowded, but you can still see correlations.
    Zoomed in scatter plot of 1000 words
    Zoomed in
    This may lag on less powerful computers, so you can either choose to lower the numbers of words shown, by changing vectors[:1000] to something more like vectors[:250], or change words to a list of your own making.
    """.splitlines()

def get_random_sentences():
    real_sentence = get_human_sentences()
    sentences = []
    vocubalary = set()
    for sentence in real_sentence:
        vocubalary.update(sentence.split())
    vocubalary = list(vocubalary)

    for sentence in real_sentence:    
        len_sentence = len(sentence.split())
        sentences.append(' '.join(np.random.choice(vocubalary, size=len_sentence)))
    return sentences


def get_ai_generated_sentences():
    return """
    Linear regression is a statistical method used for modeling the relationship between a dependent  variable and one or more independent variables by fitting a linear equation to observed data. The simplest form is simple linear regression, which deals with the relationship between two variables, while multiple linear regression deals with two or more predictors.

    Simple Linear Regression:
    Model Representation:
    In simple linear regression, we assume a linear relationship between the independent variable 
    x and the dependent variable 
    y. The linear equation is represented as:
    x is 0).
    Objective:
    The goal is to find the best-fitting line by minimizing the sum of the squared differences between the observed and predicted values.

    Method:
    The most common method for finding the optimal values for 
    b is the least squares method.
    Multiple Linear Regression:
    Model Representation:
    For multiple linear regression with 
    n independent variables, the linear equation is represented as:

    y is the dependent variable.
    are the independent variables.
    is the intercept.
    are the coefficients for the independent variables.
    Objective:
    The objective is still to minimize the sum of squared differences between the observed and predicted values.

    Method:
    The least squares method is also used in multiple linear regression.

    Key Concepts:
    Residuals:
    Residuals are the differences between the observed and predicted values. In linear regression, the goal is to minimize the sum of squared residuals.

    Assumptions:
    Linear regression assumes that there is a linear relationship between the variables, the residuals are normally distributed, and there is homoscedasticity (constant variance of residuals).

    Evaluation Metrics:
    Common metrics for evaluating the performance of a linear regression model include Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared.

    Gradient Descent:
    In addition to the least squares method, optimization algorithms like gradient descent can be used to find optimal coefficients.

    Regularization:
    Regularization techniques (L1 and L2 regularization) can be applied to prevent overfitting and improve the generalization of the model.

    Linear regression is widely used in various fields, including economics, finance, biology, and engineering, for predictive modeling and understanding the relationships between variables. It serves as a fundamental building block for more advanced machine learning techniques.
    Deep learning is a subset of machine learning that focuses on the use of neural networks to model and solve complex problems. It involves training artificial neural networks on large amounts of data to make decisions or predictions without explicit programming. Here are key concepts and components of deep learning:

    Neural Networks:
    Artificial Neurons:
    The fundamental building blocks of neural networks are artificial neurons, also called nodes or perceptrons. These neurons take input, apply weights, perform an activation function, and produce an output.

    Layers:
    Neural networks are organized into layers:

    Input Layer: Receives initial data.
    Hidden Layers: Intermediate layers between the input and output. Deep networks have multiple hidden layers.
    Output Layer: Produces the final prediction or output.
    Weights and Biases:
    Weights determine the strength of connections between neurons, and biases provide an additional input to each neuron. During training, these parameters are adjusted to minimize the difference between predicted and actual outputs.

    Training Process:
    Forward Propagation:
    During training, data is fed forward through the network to make predictions. The difference between predictions and actual values is the loss.

    Backpropagation:
    The error is then propagated backward through the network. The weights and biases are adjusted using optimization algorithms (e.g., gradient descent) to minimize the loss.

    Epochs:
    Training typically occurs over multiple passes through the entire dataset, called epochs, to improve the model's performance.

    Activation Functions:
    Sigmoid: Scales output between 0 and 1.
    ReLU (Rectified Linear Unit): Outputs the input for positive values; otherwise, outputs zero.
    TanH: Scales output between -1 and 1.
    Softmax: Used in the output layer for multi-class classification problems.
    Types of Deep Learning:
    Convolutional Neural Networks (CNNs):
    Designed for image processing and pattern recognition tasks.

    Recurrent Neural Networks (RNNs):
    Suitable for sequential data, like time series or natural language.

    Generative Adversarial Networks (GANs):
    Consists of a generator and a discriminator, used for generating realistic data.

    Applications:
    Image and Speech Recognition:
    CNNs excel in image classification, while RNNs are useful for speech recognition.

    Natural Language Processing (NLP):
    RNNs and transformers are widely used for tasks like language translation, sentiment analysis, and text generation.

    Healthcare:
    Deep learning is applied to medical imaging, diagnosis, and drug discovery.

    Autonomous Vehicles:
    Neural networks are used for object detection, path planning, and decision-making.

    Recommendation Systems:
    Deep learning is employed in recommendation engines for personalized content suggestions.

    Frameworks and Libraries:
    TensorFlow:
    Developed by Google, widely used in research and industry.

    PyTorch:
    Popular for its dynamic computation graph and is favored by researchers.

    Keras:
    High-level neural networks API that can use TensorFlow or Theano as a backend.

    Deep learning has made significant advancements in recent years, contributing to breakthroughs in various domains. Its success is attributed to the availability of large datasets, powerful hardware (GPUs), and sophisticated algorithms. As a rapidly evolving field, ongoing research continues to refine existing techniques and introduce new architectures.
        
    NLP stands for Natural Language Processing. It is a field of artificial intelligence (AI) that focuses on the interaction between computers and humans using natural language. The goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and contextually relevant.

    Key components and tasks within NLP include:

    Text Analysis:

    Tokenization: Breaking down text into words or phrases (tokens).
    Part-of-Speech Tagging (POS): Assigning grammatical labels (such as noun, verb, adjective) to each word in a sentence.
    Syntax and Grammar:

    Parsing: Analyzing the grammatical structure of sentences to understand relationships between words.
    Semantics:

    Named Entity Recognition (NER): Identifying and classifying named entities (such as names of people, organizations, locations) in text.
    Word Sense Disambiguation (WSD): Determining the meaning of a word based on context.
    Language Understanding:

    Sentiment Analysis: Determining the sentiment (positive, negative, neutral) expressed in a piece of text.
    Text Classification: Assigning predefined categories or labels to text based on its content.
    Language Generation:

    Text Generation: Creating human-like text based on given input or context.
    Summarization: Creating concise representations of longer text while preserving its key information.
    Machine Translation:

    Translating text from one language to another: Examples include Google Translate.
    Question Answering:

    Extracting answers from text: Systems that can answer questions posed in natural language.
    Dialogue Systems:

    Conversational Agents (Chatbots): Interacting with users in a natural language conversation.
    NLP is applied in various domains and applications:

    Search Engines: Improving search results by understanding user queries.
    Virtual Assistants: Enabling voice-activated virtual assistants like Siri or Alexa.
    Social Media Analysis: Analyzing sentiments, trends, and user behavior on social platforms.
    Healthcare: Extracting information from medical records or assisting in diagnostics.
    Finance: Analyzing financial reports, sentiment in financial news, and predicting market trends.
    Customer Support: Automating responses and understanding customer queries.
    NLP often involves complex tasks and challenges, as human language is inherently nuanced and context-dependent. Recent advancements in deep learning, particularly using models like transformers, have significantly improved the performance of NLP systems, leading to breakthroughs in tasks like language translation, text summarization, and question answering. Popular NLP frameworks include spaCy, NLTK, TensorFlow, and PyTorch.
    """.splitlines()
