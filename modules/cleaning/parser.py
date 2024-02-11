#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:19:34 2024

@author: yanncauchepin
"""

import re
import pandas as pd

def customized_parser_imdb(df, column_name) :
    """
    Applied to IMDB dataset.
    """

    def parser_text(text) :
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        text = (re.sub('[\W]+', ' ', text.lower()) +
                ' '.join(emoticons).replace('-', ''))
        return text

    """
    As you can see here, the text contains HTML markup as well as punctuation and
    other non-letter characters. While HTML markup does not contain many useful
    semantics, punctuation marks can represent useful, additional information in
    certain NLP contexts. However, for simplicity, we will now remove all punctuation
    marks except for emoticon characters, such as :), since those are certainly useful
    for sentiment analysis.
    Via the first regex, <[^>]*>, we tried to remove all of the HTML markup from the
    movie reviews. Although many programmers generally advise against the use of regex
    to parse HTML, this regex should be sufficient to clean this particular dataset.
    Since we are only interested in removing HTML markup and do not plan to use the
    HTML markup further, using regex to do the job should be acceptable. However, if
    you prefer to use sophisticated tools for removing HTML markup from text, you can
    take a look at Python’s HTML parser module.
    After we removed the HTML markup, we used a slightly more complex regex to find
    emoticons, which we temporarily stored as emoticons. Next, we removed all non-word
    characters from the text via the regex [\W]+ and converted the text into lowercase
    characters.
    Dealing with word capitalization. In the context of this analysis, we assume that
    the capitalization of a word—for example, whether it appears at the beginning of
    a sentence—does not contain semantically relevant information. However, note that
    there are exceptions; for instance, we remove the notation of proper names. But
    again, in the context of this analysis, it is a simplifying assumption that the
    letter case does not contain information that is relevant for sentiment analysis.
    Eventually, we added the temporarily stored emoticons to the end of the processed
    document string. Additionally, we removed the nose character (- in :-)) from the
    emoticons for consistency.
    Although the addition of the emoticon characters to the end of the cleaned document
    strings may not look like the most elegant approach, we must note that the order
    of the words doesn’t matter in our bag-of-words model if our vocabulary consists
    of only one-word tokens.
    """

    if column_name not in df.columns :
        raise ValueError(f"Column '{column_name}' is not present in the dataframe provided.\n")
    else :
        df[column_name] = df[column_name].apply(parser_text)

    return df


if __name__ == '__main__' :
    
    """EXAMPLE"""
    
    data = {
        'text': ["<html>This is a test review with <em>HTML</em> markup and some punctuation, like :) and :-)</html>",
                 "<html>:-) Another review with more <em>markup</em> and punctuation! </html>"]
        }
    df = pd.DataFrame(data)
    df_cleaned = customized_parser_imdb(df, 'text')
    print(f"- First test review :\n{df_cleaned.loc[0, 'text']}")
    print(f"- Second test review :\n{df_cleaned.loc[1, 'text']}")