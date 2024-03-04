import sys
sys.path.append('/home/yanncauchepin/Git/PublicProjects/NaturalLanguageProcessing/')

import samples.classifier_opinions.preprocessing as opinion_preprocessing

import modules.cleaning.parser as cleaning_parser
import modules.cleaning.tokenizer as cleaning_tokenizer
import modules.cleaning.stemming as cleaning_stemming

import numpy as np
import pandas as pd

if __name__ == '__main__' :

    df_amazon = opinion_preprocessing.load_dataframe("amazon")
    
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(stop_words='english')
    texts_vec = vectorizer.fit_transform(df_amazon['texts'])
    texts_vec.todense()

    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf = TfidfTransformer()
    texts_tfidf = tfidf.fit_transform(texts_vec)
    texts_tfidf = texts_tfidf.todense()
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(texts_tfidf, df_amazon['opinions'],
                                                        test_size = 0.2, random_state = 0)
    
    from sklearn.naive_bayes import MultinomialNB
    naive_bayes_classifier = MultinomialNB()
    X_train = np.asarray(X_train)
    naive_bayes_classifier.fit(X_train, y_train)
    X_test = np.asarray(X_test)
    y_pred = naive_bayes_classifier.predict(X_test)
    
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    
    from sklearn.svm import SVC
    svc_classifier = SVC(kernel='linear')
    X_train = np.asarray(X_train)
    svc_classifier.fit(X_train, y_train)
    X_test = np.asarray(X_test)
    y_pred = svc_classifier.predict(X_test)
    
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    
    import pickle
    """Python provides a very convenient way for us to do this through the pickle 
    module. Pickling in Python refers to serializing and deserializing Python 
    object structures. In other words, by using the pickle module, we can save 
    the Python objects that are created as part of model training for reuse."""
    pickle.dump(vectorizer, open('vectorizer_classifier_opinion.pickle', 'wb'))
    pickle.dump(tfidf, open('tfidf_transformer_classifier_opinion.pickle', 'wb'))
    pickle.dump(naive_bayes_classifier, open('naive_bayes_classifier_opinion.pickle', 'wb'))
    pickle.dump(svc_classifier, open('svc_classifier_opinion.pickle', 'wb'))
    
    def opinion_pred(classifier, training_matrix, doc):
        X_new = training_matrix.transform(pd.Series(doc))
        X_new = X_new.todense()
        tfidf = TfidfTransformer()
        X_new = np.asarray(X_new)
        X_tfidf_new = tfidf.fit_transform(X_new)
        X_tfidf_new = X_tfidf_new.todense()
        X_tfidf_new = np.asarray(X_tfidf_new)
        y_new = classifier.predict(X_tfidf_new)
        if y_new[0] == 0:
            return 'negative opinion'
        elif y_new[0] == 1:
            return 'positive opinion'
    
    classifier = pickle.load(open('naive_bayes_classifier_opinion.pickle', 'rb'))
    vectorizer = pickle.load(open('vectorizer_classifier_opinion.pickle', 'rb'))
    new_doc = "I do not want to stay in my accomodation since my neighbor have realized a party."
    print(opinion_pred(classifier, vectorizer, new_doc))
    