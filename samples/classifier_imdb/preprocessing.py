import os
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np
import re

class ImdbPreprocessing():
    """The movie review dataset consists of 50,000 polar movie reviews that are 
    labeled as either positive or negative; here, positive means that a movie 
    was rated with more than six stars on IMDb, and negative means that a movie 
    was rated with fewer than five stars on IMDb.
    For our own convenience, we will also store the assembled and shuffled movie 
    review dataset as a CSV file."""
    
    root_path = "/media/yanncauchepin/ExternalDisk/Datasets/NaturalLanguageProcessing/classifier_imdb/"
    input_folder_path = "imdb"
    input_csv_path = "imdb.csv"
    
    labels = {'pos': 1, 'neg': 0}
    
    def __init__(self):
        pass
    
    @staticmethod
    def load_dataframe(source="csv") :
        if source == 'csv' :
            df = pd.read_csv(os.path.join(ImdbPreprocessing.root_path, ImdbPreprocessing.input_csv_path), encoding='utf-8')
            df = df.rename(columns={"0": "review", "1": "sentiment"})
        elif source == "folder" :
            df = pd.DataFrame()
            for split in ('test', 'train'):
                for classes in ('pos', 'neg'):
                    path = os.path.join(ImbdPreprocessing.root_path, ImdbPreprocessing.input_folder_path, split, classes)
                    files = sorted(os.listdir(path))
                    for file in tqdm(files, desc=f'Processing {split}/{classes}', unit='files') :
                        with open(os.path.join(path, file), 'r', encoding='utf-8') as review:
                            txt = review.read()
                            df = df.append([[txt, ImdbPreprocessing.labels[classes]]], ignore_index=True)
            df.columns = ['review', 'sentiment']
            np.random.seed(0)
            df = df.reindex(np.random.permutation(df.index))
        else :
            raise ValueError(f"Source parameter '{source}' not recognized.\n")
        return df
    
    @staticmethod
    def text_parser(text) :
        # remove all of the HTML markup
        text = re.sub('<[^>]*>', '', text)
        # store emoticons and remove the nose character
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        emoticons = [emoticon.replace('-', '') for emoticon in emoticons]
        # remove non-word characters, convert into lower case, end with emoticons
        text = (re.sub('[\W]+', ' ', text.lower()) +' '.join(emoticons))
        return text
    
    @staticmethod
    def df_parser(df, column_name) :
        if column_name not in df.columns :
            raise ValueError(f"Column '{column_name}' is not present in the "
                             "dataframe provided.\n")
        else :
            df[column_name] = df[column_name].apply(ImdbPreprocessing.text_parser)
        return df


if __name__ == '__main__' :

    """EXAMPLE"""

    df = ImdbPreprocessing.load_dataframe(source="csv")
