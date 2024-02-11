#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:02:01 2024

@author: yanncauchepin
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



"""BAG OF WORDS"""
"""
The bag-of-words model allows us to represent text as numerical feature vectors.
The idea behind bag-of-words is quite simple and can be summarized as follows:

1.  We create a vocabulary of unique tokens—for example, words—from the entire
    set of documents.
2.  We construct a feature vector from each document that contains the counts
    of how often each word occurs in the particular document.

Since the unique words in each document represent only a small subset of all the
words in the bag-of-words vocabulary, the feature vectors will mostly consist of
zeros, which is why we call them sparse.
"""
"""N-GRAM"""
"""
The sequence of items in the bag-of-words model that we just created is also called
the 1-gram or unigram model—each item or token in the vocabulary represents a single
word. More generally, the contiguous sequences of items in NLP—words, letters,
or symbols—are also called n-grams. The choice of the number, n, in the n-gram
model depends on the particular application.
To summarize the concept of the n-gram representation, the 1-gram and 2-gram
representations of our first document, “the sun is shining”, would be constructed
as follows:
1-gram : “the”, “sun”, “is”, “shining”
2-gram : “the sun”, “sun is”, “is shining”
We could switch to a 2-gram representation by initializing a new CountVectorizer
instance with ngram_range=(2,2).
"""
"""TERM FREQUENCY INVERSE DOCUMENT FREQUENCY"""
"""
When we are analyzing text data, we often encounter words that occur across multiple
documents from both classes.
The term frequency-inverse document frequency (tf-idf) can be used to downweight
these frequently occurring words in the feature vectors. The tf-idf can be defined
as the product of the term frequency and the inverse document frequency :
tf-idf(t,d) = tf(t,d) x idf(t,d)
tf(t, d) is the term frequency count.vocabulary_
idf(t, d) is the inverse document frequency, which can be calculated as follows :
idf(t,d) = log( nd / (1 + df(d,t)) ) where :
nd is the total number of documents
df(t,d) is the number of document d that contain the term t
Note that adding the constant 1 to the denominator is optional and serves the
purpose of assigning a non-zero value to terms that occur in none of the training
examples ; the log is used to ensure that low document frequencies are not given
too much weight.
"""
"""THE WORD2VEC MODEL"""
"""
A more modern alternative to the bag-of-words model is word2vec.
The word2vec algorithm is an unsupervised learning algorithm based on neural
networks that attempts to automatically learn the relationship between words. 
The idea behind word2vec is to put words that have similar meanings into similar
clusters, and via clever vector spacing, the model can reproduce certain words
using simple vector math, for example, king – man + woman = queen.
"""

def bag_of_words(document, ngram_range=(1,1)) :
    """
    Each index position in the feature vectors shown here corresponds to the integer
    values that are stored as dictionary items in the vocabulary index.
    Values in the feature vectors are also called the raw term frequencies.
    """
    count = CountVectorizer(ngram_range=ngram_range)
    bag_of_words = count.fit_transform(document)
    return {
        "vocabulary_index" : count.vocabulary_,
        "feature_vectors" : bag_of_words.toarray()
        }    


def tfidf_bag_of_words(document, ngram_range=(1,1)) :
    count = CountVectorizer(ngram_range=ngram_range)
    tfidf = TfidfTransformer(
        use_idf=True,
        norm='l2',
        smooth_idf=True
        )
    np.set_printoptions(precision=2)
    bag_of_words = count.fit_transform(document)
    return {
        "vocabulary_index" : count.vocabulary_,
        "tfidf_vectors" : tfidf.fit_transform(bag_of_words).toarray()
        }
    

if __name__ == '__main__' :
    
    """EXAMPLE"""
    
    document = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining, the weather is sweet,'
                 'and one and one is two'])
    bag_of_words = bag_of_words(document)
    print(f"Vocabularly index :\n{bag_of_words['vocabulary_index']}")
    print(f"Feature vectors :\n{bag_of_words['feature_vectors']}")
    tfidf_bag_of_words = tfidf_bag_of_words(document)
    print(f"Vocabularly index :\n{tfidf_bag_of_words['vocabulary_index']}")
    print(f"Tfidf vectors :\n{tfidf_bag_of_words['tfidf_vectors']}")
    """
    The word 'is' had the largest term frequency in the third document, being the most
    frequently occurring word. However, after transforming the same feature vector
    into tf-idfs, the word 'is' is now associated with a relatively small tf-idf
    (0.45) in the third document, since it is also present in the first and second
    document and thus is unlikely to contain any useful discriminatory information.
    The word 'is' has a term frequency of 3 (tf = 3) in the third document, and the
    document frequency of this term is 3 since the term 'is' occurs in all three
    documents (df = 3).
    """


