#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 18:00:57 2024

@author: yanncauchepin
"""

import collection.classifier_imdb.preprocessing as imdb_preprocessing
import modules.cleaning.parser as cleaning_parser
import modules.cleaning.tokenizer as cleaning_tokenizer

if __name__ == '__main__' :
    
    """
    EXAMPLE
    """
    df_imdb = imdb_preprocessing.load_dataframe()
    df_imdb= cleaning_parser.customized_parser_imdb(df_imdb, 'review')
    imdb_tokens = cleaning_tokenizer.porter_stemming_tokenizer(df_imdb.loc[0, 'review'])
    print(cleaning_tokenizer.stop_word(imdb_tokens))
    