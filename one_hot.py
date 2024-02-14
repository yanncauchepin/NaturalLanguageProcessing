#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:39:42 2024

@author: yanncauchepin
"""


def one_hot_matrix(tokens, vocabulary_index):
    one_hot_matrix = np.zeros(len(tokens), len(vocabulary_index))
    for i, token in enumerate(tokens):
        one_hot_matrix[i][position_index[token]] = 1
    return one_hot_matrix


if __name__ == "__main__" :
    
    """EXAMPLE"""
    