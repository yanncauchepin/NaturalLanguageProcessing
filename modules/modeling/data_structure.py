import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    TfidfTransformer
    )

class DataStructure():
    
    def __init(self):
        pass

    @staticmethod
    def vocabulary(tokens) :
        set_words = set()
        for token in tokens :
            set_words.add(token)
        return set_words
    
    @staticmethod
    def vocabulary_index(tokens) :
        set_words = set()
        for token in tokens :
            set_words.add(token)
        dict_words = dict()
        for i, token in enumerate(set_words) :
            dict_words[token] = i
        return dict_words
    
    @staticmethod
    def _bag_of_words(document, vocabulary_index) :
        bow_matrix = np.zeros(len(document), len(vocabulary_index))
        for i, text in document :
            for token in text.split() :
                bow_matrix[i][vocabulary_index[token]] += 1
        return {"bow_matrix" : bow_matrix, "vocabulary_index" : vocabulary_index}


    @staticmethod
    def bag_of_words(document, ngram_range=(1,1)) :
        """The bag-of-words model allows us to represent text as numerical feature vectors.
        The idea behind bag-of-words is to create a vocabulary of unique tokens and a
        feature vector that contains the counts of how often each word occurs in the
        particular document. Feature vectors will mostly be sparse.
        The contiguous sequences of items are called n-grams. For example, the 1-gram
        and 2-gram representations of 'My name is Yann' will be :
        - 1-gram : "My", "name", "is", "Yann"
        - 2-gram : "My name", "name is", "is Yann"
        Do not take into account context, semantics or meanings associated with tokens."""
        """Each index position in the feature vectors corresponds to the integer
        values that are stored as dictionary items in the vocabulary index.
        Values in the feature vectors are also called the raw term frequencies."""
        count = CountVectorizer(
            encoding="utf-8",
            ngram_range=ngram_range,
            strip_accents=None,
            lowercase=True,
            stop_words=None,
            analyzer="word",
            max_df=1.0,
            min_df=1,
            max_features=None,
            vocabulary=None,
            binary=False
            )
        bag_of_words = count.fit_transform(document)
        return {
            "vocabulary_index" : count.vocabulary_,
            "feature_vectors" : bag_of_words
            }

    @staticmethod
    def tfidf_bag_of_words(document, ngram_range=(1,1)) :
        """The term frequency-inverse document frequency (tf-idf) can be used to downweight
        the frequently occurring words in the feature vectors. The tf-idf can be defined
        as the product of the term frequency and the inverse document frequency :
        tf-idf(t,d) = tf(t,d) x idf(t,d)
        - tf(t, d) is the term frequency in a document, count.vocabulary_ here in the code.
        A well process is to normalize it by dividing it with the count of terms in the
        document.
        - idf(t, d) is the inverse document frequency that measures the importance of
        a term in a document, which can be calculated as follows :
        idf(t,d) = log( nd / (1 + df(d,t)) ) where :
        - nd is the total number of documents.
        - df(t,d) is the number of document d that contain the term t.
        - addition plus one in the denominator is a simple choice which is not always
        applied in order to avoid division by zero. Corresponds to parameter 'smooth_idf'
        in TfidfTransformer.
        The pattern of information carried across terms that are rarely present but
        carry a high amount of information is better handle by tf-idf.
        Each tfidf row or vector is normalized to have a unit norm :
        - l2, the sum of squares of the vector elements is equal to 1.
        - l1, the sum of absolute values of the vector elements is 1.
        Do not take into account context, semantics or meanings associated with tokens."""
        count = CountVectorizer(
            encoding="utf-8",
            ngram_range=ngram_range,
            strip_accents=None,
            lowercase=True,
            stop_words=None,
            analyzer="word",
            max_df=1.0,
            min_df=1,
            max_features=None,
            vocabulary=None,
            binary=False
            )
        bag_of_words = count.fit_transform(document)
        tfidf = TfidfTransformer(
            norm='l2',
            use_idf=True,
            smooth_idf=True
            )
        tfidf_bag_of_words = tfidf.fit_transform(bag_of_words)
        return {
            "vocabulary_index" : count.vocabulary_,
            "tfidf_vectors" : tfidf_bag_of_words
            }
    
    @staticmethod
    def tfidf_bag_of_words_alternative(document, ngram_range=(1,1)) :
        count = TfidfVectorizer(
            encoding="utf-8",
            ngram_range=ngram_range,
            strip_accents=None,
            lowercase=True,
            stop_words=None,
            analyzer="word",
            max_df=1.0,
            min_df=1,
            max_features=None,
            vocabulary=None,
            binary=False,
            norm='l2',
            use_idf=True,
            smooth_idf=True
            )
        tfidf_bag_of_words = count.fit_transform(document)
        return {
            "vocabulary_index" : count.vocabulary_,
            "tfidf_vectors" : tfidf_bag_of_words
            }
    
    @staticmethod
    def tfidf_feature_vectors(feature_vectors) :
        tfidf = TfidfTransformer(
            norm='l2',
            use_idf=True,
            smooth_idf=True
            )
        tfidf_bag_of_words = tfidf.fit_transform(feature_vectors)
        return tfidf_bag_of_words



if __name__ == '__main__' :

    """EXAMPLE"""

    document = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining, the weather is sweet,'
                 'and one and one is two'])
    bag_of_words = DataStructure.bag_of_words(document)
    print(f"Vocabularly index :\n{bag_of_words['vocabulary_index']}")
    print(f"Feature vectors :\n{bag_of_words['feature_vectors']}")
    print(f"Shape 0 :\n{bag_of_words['feature_vectors'].shape[0]}")
    tfidf_bag_of_words = DataStructure.tfidf_bag_of_words(document)
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
