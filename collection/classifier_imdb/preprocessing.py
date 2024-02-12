#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:02:01 2024

@author: yanncauchepin
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np

"""The movie review dataset consists of 50,000 polar movie reviews that are labeled
as either positive or negative; here, positive means that a movie was rated with
more than six stars on IMDb, and negative means that a movie was rated with fewer
than five stars on IMDb.
For our own convenience, we will also store the assembled and shuffled movie review
dataset as a CSV file."""

root_path = "/media/yanncauchepin/ExternalDisk/Datasets/NaturalLanguageProcessing/classifier_imdb/"
input_folder_path = "imdb"
input_csv_path = "imdb.csv"

labels = {'pos': 1, 'neg': 0}


def load_dataframe(source="csv") :
    if source == 'csv' :
        df = pd.read_csv(os.path.join(root_path, input_csv_path), encoding='utf-8')
        df = df.rename(columns={"0": "review", "1": "sentiment"})
    elif source == "folder" :
        df = pd.DataFrame()
        for split in ('test', 'train'):
            for classes in ('pos', 'neg'):
                path = os.path.join(root_path, input_folder_path, split, classes)
                files = sorted(os.listdir(path))
                for file in tqdm(files, desc=f'Processing {split}/{classes}', unit='files') :
                    with open(os.path.join(path, file), 'r', encoding='utf-8') as review:
                        txt = review.read()
                        df = df.append([[txt, labels[classes]]], ignore_index=True)
        df.columns = ['review', 'sentiment']
        np.random.seed(0)
        df = df.reindex(np.random.permutation(df.index))
    else :
        raise ValueError(f"Source parameter '{source}' not recognized.\n")
    return df


if __name__ == '__main__' :
    
    """EXAMPLE"""
    
    df = load_dataframe(source="folder")