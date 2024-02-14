#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:19:34 2024

@author: yanncauchepin
"""

import re
import pandas as pd


def imdb_text_parser(text) :
    """To test"""
    # remove all of the HTML markup
    text = re.sub('<[^>]*>', '', text)
    # store emoticons and remove the nose character
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    emoticons = [emoticon.replace('-', '') for emoticon in emoticons]
    # remove non-word characters, convert into lower case, end with emoticons
    text = (re.sub('[\W]+', ' ', text.lower()) +' '.join(emoticons))
    return text


def imdb_df_parser(df, column_name) :
    if column_name not in df.columns :
        raise ValueError(f"Column '{column_name}' is not present in the dataframe provided.\n")
    else :
        df[column_name] = df[column_name].apply(imdb_text_parser)
    return df


from bs4 import BeautifulSoup
"""BeautifulSoup module with the HTML Parser parameter is a HTML parser object.
An example could be 'soup = BeautifulSoup(html_text, "html.parser")'."""
def html_beautiful_soup_parser(html) :
    return BeautifulSoup(html).get_text()


if __name__ == '__main__' :
    
    """EXAMPLE"""
    
    data = {
        'text': ["<html>This is a test review with <em>HTML</em> markup and some punctuation, like :) and :-)</html>",
                 "<html>:-) Another review with more <em>markup</em> and punctuation! </html>"]
        }
    df = pd.DataFrame(data)
    df_cleaned = imdb_df_parser(df, 'text')
    print(f"- First test review :\n{df_cleaned.loc[0, 'text']}")
    print(f"- Second test review :\n{df_cleaned.loc[1, 'text']}")
    
    html = "<!DOCTYPE html><html><body><h1>My First Heading</h1><p>My first paragraph.</p></body></html>"