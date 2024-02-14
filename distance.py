#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 19:51:51 2024

@author: yanncauchepin
"""

import numpy as np


"""The value of cosine similarity would lie in the range -1 to +1. The value +1 
indicates that the vectors are perfectly similar, and the value -1 indicates 
that the vectors are perfectly dissimilar or exactly opposite to each other."""
def cosine_similarity(vector1, vector2) :
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    product = np.dot(vector1, vector2)
    magnitude1 = np.sqrt(np.sum(vector1**2))
    magnitude2 = np.sqrt(np.sum(vector2**2))
    return product/(magnitude1*magnitude2)


def print_distance_document(bow_matrix_array) :
    nb_text = len(bow_matrix_array)
    for i in range(nb_text):
        for j in range(1+i, nb_text):
            distance = cosine_similarity(bow_matrix_array[i], bow_matrix_array[j])
            print(f"Cosine similarity between text {i} and {j} is : {distance}\n")


def word_mover_distance():
    pass


if __name__ == '__main__' :
    
    """EXAMPLE"""
    
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