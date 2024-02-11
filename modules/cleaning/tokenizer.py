#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:37:28 2024

@author: yanncauchepin
"""

import nltk # Natural Language Toolkit

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
"""
The Porter stemming algorithm is probably the oldest and simplest stemming algorithm.
Other popular stemming algorithms include the newer Snowball stemmer (Porter2 or
English stemmer) and the Lancaster stemmer (Paice/Husk stemmer). While both the
Snowball and Lancaster stemmers are faster than the original Porter stemmer, the
Lancaster stemmer is also notorious for being more aggressive than the Porter
stemmer, which means that it will produce shorter and more obscure words. These
alternative stemming algorithms are also available through the NLTK package.
While stemming can create non-real words, such as 'thu' (from 'thus'), as shown
in the previous example, a technique called lemmatization aims to obtain the
canonical (grammatically correct) forms of individual words—the so-called lemmas.
However, lemmatization is computationally more difficult and expensive compared
to stemming and, in practice, it has been observed that stemming and lemmatization
have little impact on the performance of text classification.
"""

nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
"""
Stop words are simply those words that are extremely common in all sorts of texts
and probably bear no (or only a little) useful information that can be used to
distinguish between different classes of documents. Examples of stop words are
is, and, has, and like. Removing stop words can be useful if we are working with
raw or normalized term frequencies rather than tf-idfs, which already downweight
the frequently occurring words.
To remove stop words from the movie reviews, we will use the set of 127 English
stop words that is available from the NLTK library, which can be obtained by calling
the nltk.download function.
"""


def simple_tokenizer(text):
    return text.split()


def porter_stemming_tokenizer(text) :
    """
    Word stemming is the process of transforming a word into its root form. 
    It allows us to map related words to the same stem.
    """
    return [porter.stem(word) for word in text.split()]


def stop_word(tokens) :
    return [token for token in tokens if token not in stop]


if __name__ == '__main__' :
    
    """EXAMPLE"""

    text = "This is a test sentence for tokenization."
    tokens = simple_tokenizer(text)
    print("Tokenization output:", tokens)
    
    text = "running runs ran"
    stemmed_tokens = porter_stemming_tokenizer(text)
    print("Stemming output:", stemmed_tokens)
    
    tokens = ["This", "is", "a", "test", "sentence", "with", "some", "stop", "words"]
    tokens_without_stopwords = stop_word(tokens)
    print("Tokens without stopwords:", tokens_without_stopwords)
    
    