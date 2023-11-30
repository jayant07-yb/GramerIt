import numpy as np

class GloveVectorizer():
    def __init__(self, path='weights/glove.6B.50d.txt'):
        embeddings_dict = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
        
        self.embeddings_dict = embeddings_dict
        self.embedding_dim = len(vector)
    
    def transform_word(self, word):
        if word not in self.embeddings_dict.keys():
            return np.zeros(self.embedding_dim)
        return self.embeddings_dict[word]
