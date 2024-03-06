import numpy as np
import gensim
from gensim.models import KeyedVectors


class Distance():
    
    def __init__(self):
        pass
    
    @staticmethod
    def cosine_similarity(vector1, vector2) :
        """The value of cosine similarity would lie in the range -1 to +1. The value +1
        indicates that the vectors are perfectly similar, and the value -1 indicates
        that the vectors are perfectly dissimilar or exactly opposite to each other."""
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)
        product = np.dot(vector1, vector2)
        magnitude1 = np.sqrt(np.sum(vector1**2))
        magnitude2 = np.sqrt(np.sum(vector2**2))
        return product/(magnitude1*magnitude2)

    @staticmethod
    def print_distance_document(bow_matrix_array) :
        nb_text = len(bow_matrix_array)
        for i in range(nb_text):
            for j in range(1+i, nb_text):
                distance = cosine_similarity(bow_matrix_array[i], bow_matrix_array[j])
                print(f"Cosine similarity between text {i} and {j} is : {distance}\n")


    @staticmethod
    def word_mover_distance():
        """Word Mover's Distance (WMD) is more relevant than cosine similarity,
        especially to measure the distance for documents on word embeddings. It defines
        the dissimilarity between two text documents as the minimum amount of distance
        that the embedded words of one document need to travel to reach the embedded
        words of another document.
        WMD computes the pairwise Euclidean distance between words across the sentences
        and it defines the distance between two documents as the minimum cumulative cost
        in terms of the Euclidean distance required to move all the words from the first
        sentence to the second sentence."""
        pass
        """
        model=KeyedVectors.load_word2vec_format('/Users/amankedia/Desktop/Sunday/nlp-book/Chapter 5/Code/GoogleNews-vectors-negative300.bin', binary=True)
        sentence_1 = "Obama speaks to the media in Illinois"
        sentence_2 = "President greets the press in Chicago"
        sentence_3 = "Apple is my favorite company"
        word_mover_distance = model.wmdistance(sentence_1, sentence_2)
        word_mover_distance
        """
        """Normalize word embeddings"""
        """
        model.init_sims(replace = True)
        """


if __name__ == '__main__' :

    """EXAMPLE"""
    '''
    bow_matrix_array = [[0, 1, 0, 1, 1, 0, 1, 0, 0],
                        [0, 1, 0, 0, 0, 1, 1, 0, 1],
                        [2, 3, 2, 1, 1, 1, 2, 1, 1]]
    print_distance_document(bow_matrix_array)
    tfidf_bow_matrix_array = [[0., 0.43370786, 0., 0.55847784, 0.55847784, 0.,
                               0.43370786, 0., 0.],
                              [0., 0.43370786, 0., 0., 0., 0.55847784, 0.43370786,
                               0., 0.55847784],
                              [0.50238645, 0.44507629, 0.50238645, 0.19103892,
                               0.19103892, 0.19103892, 0.29671753, 0.25119322,
                               0.19103892]]
    print_distance_document(tfidf_bow_matrix_array)
    '''
    
    