'''Issues'''

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Dense, Dropout, Input
from keras.utils import np_utils

import samples.classifier_questions.preprocessing as questions_preprocessing

def text_clean(corpus, keep_list):
    '''
    Purpose : Function to keep only alphabets, digits and certain words (punctuations, qmarks, tabs etc. removed)
    
    Input : Takes a text corpus, 'corpus' to be cleaned along with a list of words, 'keep_list', which have to be retained
            even after the cleaning process
    
    Output : Returns the cleaned text corpus
    
    '''
    cleaned_corpus = pd.Series()
    for row in corpus:
        qs = []
        for word in row.split():
            if word not in keep_list:
                p1 = re.sub(pattern='[^a-zA-Z0-9]',repl=' ',string=word)
                p1 = p1.lower()
                qs.append(p1)
            else : qs.append(word)
        cleaned_corpus = pd.concat([cleaned_corpus, pd.Series(' '.join(qs))])
    return cleaned_corpus

def stopwords_removal(corpus):
    wh_words = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']
    stop = set(stopwords.words('english'))
    for word in wh_words:
        stop.remove(word)
    corpus = [[x for x in x.split() if x not in stop] for x in corpus]
    return corpus

def lemmatize(corpus):
    lem = WordNetLemmatizer()
    corpus = [[lem.lemmatize(x, pos = 'v') for x in x] for x in corpus]
    return corpus

def stem(corpus, stem_type = None):
    if stem_type == 'snowball':
        stemmer = SnowballStemmer(language = 'english')
        corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    else :
        stemmer = PorterStemmer()
        corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    return corpus

def preprocess(corpus, keep_list=[], cleaning = True, stemming = False, stem_type = None, lemmatization = False, remove_stopwords = True):
    '''
    Purpose : Function to perform all pre-processing tasks (cleaning, stemming, lemmatization, stopwords removal etc.)
    
    Input : 
    'corpus' - Text corpus on which pre-processing tasks will be performed
    'keep_list' - List of words to be retained during cleaning process
    'cleaning', 'stemming', 'lemmatization', 'remove_stopwords' - Boolean variables indicating whether a particular task should 
                                                                  be performed or not
    'stem_type' - Choose between Porter stemmer or Snowball(Porter2) stemmer. Default is "None", which corresponds to Porter
                  Stemmer. 'snowball' corresponds to Snowball Stemmer
    
    Note : Either stemming or lemmatization should be used. There's no benefit of using both of them together
    
    Output : Returns the processed text corpus
    
    '''
    
    if cleaning == True:
        corpus = text_clean(corpus, keep_list)
    
    if remove_stopwords == True:
        corpus = stopwords_removal(corpus)
    else :
        corpus = [[x for x in x.split()] for x in corpus]
    
    if lemmatization == True:
        corpus = lemmatize(corpus)
        
        
    if stemming == True:
        corpus = stem(corpus, stem_type)
    
    corpus = [' '.join(x) for x in corpus]        

    return corpus

if __name__ == '__main__' :
    
    dataset = questions_preprocessing.load_dataframe()
    dataset.pop('QType')
    dataset.pop('QType-Fine')
    
    classes = np.unique(np.array(dataset['QType-Coarse']))
    print(classes)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(pd.Series(dataset['QType-Coarse'].to_list()))
    y = label_encoder.transform(dataset['QType-Coarse'].values).astype(int)
    print(y)
    
    X = pd.Series(dataset.Question.tolist()).astype(str)
    X = preprocess(X, remove_stopwords=True)
    print(X)
    
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    tfidf = TfidfTransformer()
    X = tfidf.fit_transform(X)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                        random_state = 0)
    
    print(X_train.shape[0])
    print(X_test.shape[0])
    print(X_train[0])
    
    y_train = np_utils.to_categorical(y_train, num_classes=len(classes))
    y_test = np_utils.to_categorical(y_test, num_classes=len(classes))
    
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.3))
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()
    
    X_train_sparse = tf.sparse.SparseTensor(
        indices=X_train.nonzero(),
        values=X_train.data,
        dense_shape=X_train.shape
    )

    X_test_sparse = tf.sparse.SparseTensor(
        indices=X_test.nonzero(),
        values=X_test.data,
        dense_shape=X_test.shape
    )
    
    training_history = model.fit(X_train_sparse, y_train, epochs=10, batch_size=100)
    
    loss, accuracy = model.evaluate(X_test_sparse, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    """Let's save the model architecture and weights using the following code block"""
    import h5py
    model_structure = model.to_json()
    with open("question_classification_model.json", "w") as json_file:
        json_file.write(model_structure)
    model.save_weights("question_classification_weights.h5")
