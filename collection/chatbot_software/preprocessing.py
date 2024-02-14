#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 21:38:40 2024

@author: yanncauchepin
"""

import os
import ast

root_path = "/media/yanncauchepin/ExternalDisk/Datasets/NaturalLanguageProcessing/chatbot_software/"
input_json_path = "qa_Software.json"

def load_dataframe() :
    
    questions = list()
    answers = list()
    
    with open(os.path.join(root_path, input_json_path),'r') as input_file :
        for line in input_file :
            data = ast.literal_eval(line)
            questions.append(data['question'])
            answers.append(data['answer'])
    
    return {"questions" : questions, "answers" : answers}


if __name__ == '__main__' :
    
    """EXAMPLE"""
    
    df_software = load_dataframe()
    questions = df_software['questions']
    anwsers = df_software['answers']
    
    print(len(questions))