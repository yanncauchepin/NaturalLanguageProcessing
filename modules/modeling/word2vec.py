#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 08:35:35 2024

@author: yanncauchepin
"""

import numpy as np

"""Word2vec is a modern alternative to the bag-of-words model.
The word2vec algorithm is an unsupervised learning algorithm based on neural
networks that attempts to automatically learn the relationship between words. 
The idea behind word2vec is to put words that have similar meanings into similar
clusters, and via clever vector spacing, the model can reproduce certain words
using simple vector math, for example, king – man + woman = queen.
The output is a matrix of shape |V|*D where |V| is the size of the vocabulary
and D is the number of dimensions used to represent each word vector. Generally,
D takes values between 50 and 300 in real-life use cases.
Word2vec models can be trained by two approches :
- Predicting the context word using the target word as input, which is referred 
to as the Skip-gram method.
- Predicting the target word using the context words as input, which is referred 
to as the Continuous Bag-of-Words (CBOW) method."""
def word2vec() :
    pass


"""The Skip-gram method of Word2vec takes several parameters to predict the
context word for a target word set in input:
- input vector corresponding to the one_hot vector of with a size of the lenght
of the vocabulary. The only value 1 is set to the position of the target word.
- embediding matrix with a size of lenght of vocabulary and the number of dimensions
to represent each word vector with. Can be instantiated with random numbers.
- context matrix with a identical size than embedding matrix.
- window_size corresponding to the size of the neighborhood of to consider to 
get context words. For example with a value of 5, the window size select the 
two words before and after the target word. In case the target word is the first
or last token, non of the words is selected respectively before and after.
The output vector is yield thanks to the dot of context matrix and the intermediate 
vector where the intermediate vector is itself the dot of the input vector and 
the embedding matrix. The idea is the intermediate vector activate the context
word's entry in the context matrix. The output vector with a size of the lenght
of the vocabulary represents the chances of the word corresponding to that
index being the context word predicted by the model.
To normalize the output vector, the uses is to apply a softmax function to 
transform each values between 0 and 1 and to set their sum up to 1.
By a process of loss calculation and backpropagation, we could train the model
on labeled data and update both embedding and context matrix."""
def softmax(vector):
    return np.exp(vector)/np.sum(np.exp(vector))


"""The CBOW method is very similar to the Skip-gram method but here we try to 
predict the target word from the context word as input."""


"""To faster compute Word2vec, several strategies are available such as:
- Subsampling where the algorithm do not consider words in the neighborhood where
they occur too frequently and do not add much information. It delete those 
words and reduced the training data size. The sample rate corresponding to the
probability of keeping a word is calculated by:
proba(word_i) = (sqrt( f(word_i)/0.001 ) +1) * 0.001/f(word_i) where 
f(word_i) is the fraction of total words in the corpus.
- Negative sampling consider the same idea and involve another probability of
keeping a word by:
proba(word_i) = freq(word_i) / sum_j( freq(word_j) ) where
freq(word_j) is the number of time the word j occurs in the corpus."""



import gensim
"""Gensim library provides various method to use the pretrained model Word2vec 
directly or to fine-tine it."""


from gensim.test.utils import lee_corpus_list
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
"""Load the pretrained vectors form the pretrained Word2vec model."""
"""
model = Word2Vec(lee_corpus_list, vector_size=24, epochs=100)
word_vectors = model.wv
word_vectors.save('vectors.kv')
reloaded_word_vectors = KeyedVectors.load('vectors.kv')
"""


"""min_count parameter helps create custom vocabulary based on the text itself.
It sets a minimum threshold so that vectors are built only for words that occur 
more often than the value specified in the min_count parameter."""
sentences = [["I", "am", "trying", "to", "understand", "Natural", 
              "Language", "Processing"],
            ["Natural", "Language", "Processing", "is", "fun", 
             "to", "learn"],
            ["There", "are", "numerous", "use", "cases", "of", 
             "Natural", "Language", "Processing"]]
model = Word2Vec(sentences, min_count=1)
"""The default vector size of Word2vec is 100, but it is configurable."""
model.vector_size
"""Size of the built vocabulary"""
len(model.wv.key_to_index)
"""Built vocabulary"""
list(mdel.wv.key_to_index.keys())


if __name__ == '__main__' :
    
    """EXAMPLE"""
    