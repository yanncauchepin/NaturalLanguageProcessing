
import pandas as pd
import os
import sys
import numpy as np

"""DATASET IMDB"""

df = pd.read_csv('/home/yanncauchepin/PrivateProjects/ArtificialIntelligence/Datasets/NLP/NLP_IMDB/movie_data.csv', encoding='utf-8')
df = df.rename(columns={"0": "review", "1": "sentiment"})

"""PREPROCESSING DATA"""

import re

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

df['review'] = df['review'].apply(preprocessor)

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')


"""LOGISTIC REGRESSION SCIKITLEARN"""

"""PARTITION TRAINING AND TESTING"""
"""
we will train a logistic regression model to classify the movie reviews into
positive and negative reviews based on the bag-of-words model. First, we will
divide the DataFrame of cleaned text documents into 25,000 documents for training
and 25,000 documents for testing.
"""
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

"""GRIDSEARCHCV"""
"""
we will use a GridSearchCV object to find the optimal set of parameters for our
logistic regression model using 5-fold stratified cross-validation.

Note that for the logistic regression classifier, we are using the LIBLINEAR solver
as it can perform better than the default choice ('lbfgs') for relatively large
datasets.
"""
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)
small_param_grid = [
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [None],
        'vect__tokenizer': [tokenizer, tokenizer_porter],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    },
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [tokenizer],
        'vect__use_idf':[False],
        'vect__norm':[None],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    },
]
lr_tfidf = Pipeline([
    ('vect', tfidf),
    ('clf', LogisticRegression(solver='liblinear'))
])
gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid,
                           scoring='accuracy', cv=5,
                           verbose=2, n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)
"""
When we initialized the GridSearchCV object and its parameter grid using the
preceding code, we restricted ourselves to a limited number of parameter combinations,
since the number of feature vectors, as well as the large vocabulary, can make
the grid search computationally quite expensive. Using a standard desktop computer,
our grid search may take 5-10 minutes to complete.
In the previous code example, we replaced CountVectorizer and TfidfTransformer
from the previous subsection with TfidfVectorizer, which combines CountVectorizer
with the TfidfTransformer. Our param_grid consisted of two parameter dictionaries.
In the first dictionary, we used TfidfVectorizer with its default settings
(use_idf=True, smooth_idf=True, and norm='l2') to calculate the tf-idfs; in the
second dictionary, we set those parameters to use_idf=False, smooth_idf=False,
and norm=None in order to train a model based on raw term frequencies. Furthermore,
for the logistic regression classifier itself, we trained models using L2
regularization via the penalty parameter and compared different regularization
strengths by defining a range of values for the inverse-regularization parameter
C. As an optional exercise, you are also encouraged to add L1 regularization to
the parameter grid by changing 'clf__penalty': ['l2'] to 'clf__penalty' :
['l2', 'l1'].
"""
print(f'Best parameter set: {gs_lr_tfidf.best_params_}')

"""N-FOLD CROSS VALIDATION"""
"""
Using the best model from this grid search, let’s print the average 5-fold
cross-validation accuracy scores on the training dataset and the classification
accuracy on the test dataset.
"""
print(f'CV Accuracy: {gs_lr_tfidf.best_score_:.3f}')
clf = gs_lr_tfidf.best_estimator_
print(f'Test Accuracy: {clf.score(X_test, y_test):.3f}')
"""
The results reveal that our machine learning model can predict whether a movie
review is positive or negative with 90 percent accuracy.
"""

"""THE NAIVE BAYES CLASSIFIER"""
"""
A still very popular classifier for text classification is the naïve Bayes classifier,
which gained popularity in applications of email spam filtering. Naïve Bayes
classifiers are easy to implement, computationally efficient, and tend to perform
particularly well on relatively small datasets compared to other algorithms.
"""
