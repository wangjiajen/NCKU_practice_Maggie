#In[]
from keras.preprocessing.text import Tokenizer
import gensim
import pandas as pd
import numpy as np
from itertools import chain

#In[]
embeddings_index = dict()
w2v = gensim.models.Word2Vec.load("wiki-lemma-100D")
vocab = w2v.wv.key_to_index    
t = Tokenizer()

vocab_size = len(all_words) + 1
t.fit_on_texts(all_words)

def get_weight_matrix():
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, w2v.vector_size))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for i in range(len(all_words)):
        weight_matrix[i + 1] = w2v[all_words[i]]
    return weight_matrix

#In[]
import numpy as np
# print(model.wv['sentence'])
embedding_vectors = get_weight_matrix()
emb_layer = Embedding(vocab_size, output_dim=w2v.vector_size, weights=[embedding_vectors], input_length=FIXED_LENGTH, trainable=False)
#In[]
from transformer import TransformerBlock, TokenAndPositionEmbedding
# model.layers[0].set_weights([embeddings_index])
# model.layers[0].trainable = False
# %%


# # load embedding as a dict
# def load_embedding(filename):
#     # load embedding into memory, skip first line
#     file = open(filename,'r')
#     lines = file.readlines()[1:]
#     file.close()
#     # create a map of words to vectors
#     embedding = dict()
#     for line in lines:
#         parts = line.split()
#         # key is string word, value is numpy array for vector
#         embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
#     return embedding

# # create a weight matrix for the Embedding layer from a loaded embedding
# def get_weight_matrix(embedding, vocab):
#     # total vocabulary size plus 0 for unknown words
#     vocab_size = len(vocab) + 1
#     # define weight matrix dimensions with all 0
#     weight_matrix = np.zeros((vocab_size, 100))
#     # step vocab, store vectors using the Tokenizer's integer mapping
#     for word, i in vocab.items():
#         weight_matrix[i] = embedding.get(word)
#     return weight_matrix

# # load embedding from file
# raw_embedding = load_embedding('wiki.txt')
# # get vectors in the right order
# embedding_vectors = get_weight_matrix(raw_embedding, t.word_index)

# %%
