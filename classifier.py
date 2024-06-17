import numpy as np

import samples
from samples.classifier_imdb.preprocessing import ImdbPreprocessing
from samples.classifier_opinions.preprocessing import OpinionsPreprocessing

import modules
from modules.cleaning.tokenizer import Tokenizer
from modules.cleaning.stemmer import Stemmer
from modules.cleaning.stop_words_filter import StopWordsFilter
from modules.modeling.data_structure import DataStructure
from modules.modeling.split import Split
from modules.utils.metric import Metric


class ImdbClassifier():
    
    def __init__(self):
        df_imdb = ImdbPreprocessing.load_dataframe()
        df_imdb= ImdbPreprocessing.df_parser(df_imdb, 'review')

        imdb_tokens = Tokenizer.split(df_imdb.loc[0, 'review'])
        imdb_tokens = Stemmer.porter(imdb_tokens)
        imdb_tokens = StopWordsFilter.english(imdb_tokens)
        print(imdb_tokens)
        

class AmazonClassifier():
    
    def __init__(self):
        df_amazon = OpinionsPreprocessing.load_dataframe("amazon")
        bag_of_words = DataStructure.bag_of_words(df_amazon['texts'])
        feature_vectors = bag_of_words['feature_vectors']
        tfidf_feature_vectors = DataStructure.tfidf_feature_vectors(feature_vectors)
        dataset = Split.standard(tfidf_feature_vectors, df_amazon['opinions'], 0.2, df_amazon['opinions'])
        
        X_train = dataset['X_train'].toarray()
        X_test = dataset['X_test'].toarray()
        y_train = dataset['y_train']
        y_test = dataset['y_test']
        
        from sklearn.naive_bayes import MultinomialNB
        naive_bayes_classifier = MultinomialNB()
        naive_bayes_classifier.fit(X_train, y_train)
        y_pred = naive_bayes_classifier.predict(X_test)
        print(Metric.confusion_matrix(y_pred, y_pred))
        
        from sklearn.svm import SVC
        svc_classifier = SVC(kernel='linear')
        svc_classifier.fit(X_train, y_train)
        y_pred = svc_classifier.predict(X_test)
        print(Metric.confusion_matrix(y_pred, y_test))
        
        def pred(type_classifier, text):
            X_new = DataStructure.bag_of_words(df_amazon['texts'])['feature_vectors']
            X_new =  DataStructure.tfidf_feature_vectors(X_new)
            X_new = X_new.toarray()
            if type_classifier == 'naive_bayes':    
                y_new = naive_bayes_classifier.predict(X_new)
            elif type_classifier == 'svc':
                y_new = svc_classifier.predict(X_new)
            if y_new[0] == 0:
                return 'negative opinion'
            elif y_new[0] == 1:
                return 'positive opinion'
        
        new_doc = "I do not want to stay in my accomodation since my neighbor "\
        "have realized a party."
        print(opinion_pred('svc', new_doc))
        
    
if __name__ == '__main__':
    AmazonClassifier()
