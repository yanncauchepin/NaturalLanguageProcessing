import numpy as np

import samples.chatbot_software.preprocessing as software_preprocessing

import modules.cleaning.parser as cleaning_parser
import modules.cleaning.tokenizer as cleaning_tokenizer

"""Deprecated"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer


if __name__ == '__main__' :

    df_software = software_preprocessing.load_dataframe()
    questions = df_software['questions']
    answers = df_software['answers']

    vectorizer = CountVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(questions)
    tfidf = TfidfTransformer(norm='l2')
    X_tfidf = tfidf.fit_transform(X_vec)

    def conversation(im):
        global tfidf, answers, X_tfidf
        Y_vec = vectorizer.transform(im)
        Y_tfidf = tfidf.fit_transform(Y_vec)
        angle = np.rad2deg(np.arccos(max(cosine_similarity(Y_tfidf, \
                           X_tfidf)[0])))
        if angle > 60 :
            return "sorry, I did not quite understand that"
        else:
            return answers[np.argmax(cosine_similarity(Y_tfidf, X_tfidf)[0])]

    usr = input("Please enter your username: ")
    print("support: Hi, welcome to Q&A support. How can I help you?")
    while True:
        im = input("{}: ".format(usr))
        if im.lower() == 'bye':
            print("Q&A support: bye!")
            break
        else:
            print("Q&A support: "+conversation([im]))
