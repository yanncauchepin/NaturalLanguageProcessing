#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 21:13:22 2024

@author: yanncauchepin
"""

import collection.chatbot_electronics.preprocessing as electronics_preprocessing

import modules.cleaning.parser as cleaning_parser
import modules.cleaning.tokenizer as cleaning_tokenizer



if __name__ == '__main__' :
    
    df_electronics = electronics_preprocessing.load_dataframe()
    questions = df_electronics['questions']
    anwsers = df_electronics['answers']
    
    print(questions)
    