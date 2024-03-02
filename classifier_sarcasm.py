import os
import pandas as pd
import numpy as np
import re
import json
import gensim
import math
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import KeyedVectors
import keras
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D
import h5py



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



def parse_data(file):
    for l in open(file,'r'):
        yield json.loads(l)

root_path = "/media/yanncauchepin/ExternalDisk/Datasets/NaturalLanguageProcessing/classifier_sarcasm/"
input_path = "sarcasm/Sarcasm_Headlines_Dataset_v2.json"
dataset_path = os.path.join(root_path, input_path)


data = list(parse_data(dataset_path))
df = pd.DataFrame(data)
df.pop('article_link')
headlines = preprocess(df['headline'], lemmatization=True, remove_stopwords=True)

from gensim.models import KeyedVectors

import wget
url = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
filename = wget.download(url)
f_in = gzip.open('GoogleNews-vectors-negative300.bin.gz', 'rb')
f_out = open('GoogleNews-vectors-negative300.bin', 'wb')
f_out.writelines(f_in)

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

#Standardization
MAX_LENGTH = 10
VECTOR_SIZE = 300

def vectorize_data(data):
 
    vectors = []
 
    padding_vector = [0.0] * VECTOR_SIZE
 
    for i, data_point in enumerate(data):
 
         data_point_vectors = []
         count = 0
      
         tokens = data_point.split()
      
         for token in tokens:
             if count >= MAX_LENGTH:
                 break
             if token in model.wv.vocab:
                 data_point_vectors.append(model.wv[token])
             count = count + 1
      
    if len(data_point_vectors) < MAX_LENGTH:
         to_fill = MAX_LENGTH - len(data_point_vectors)
         for _ in range(to_fill):
             data_point_vectors.append(padding_vector)
  
    vectors.append(data_point_vectors)
 
    return vectors

vectorized_headlines = vectorize_data(headlines)

for i, vec in enumerate(vectorized_headlines): 
    if len(vec) != MAX_LENGTH:    
        print(i)
    
X_train = vectorized_headlines[:train_div]
y_train = df['is_sarcastic'][:train_div]
X_test = vectorized_headlines[train_div:]
y_test = df['is_sarcastic'][train_div:]
print('The size of X_train is:', len(X_train),
      '\nThe size of y_train is:', len(y_train),
      '\nThe size of X_test is:', len(X_test),
      '\nThe size of y_test is:', len(y_test)) 


FILTERS=8
KERNEL_SIZE=3
HIDDEN_LAYER_1_NODES=10
HIDDEN_LAYER_2_NODES=5
DROPOUT_PROB=0.35
NUM_EPOCHS=10
BATCH_SIZE=50

model = Sequential()
model.add(Conv1D(
        FILTERS, 
        KERNEL_SIZE, 
        padding='same',                                       
        strides=1, 
        activation='relu', 
        input_shape = (MAX_LENGTH, VECTOR_SIZE)
        ))
model.add(GlobalMaxPooling1D())
model.add(Dense(HIDDEN_LAYER_1_NODES, activation='relu'))
model.add(Dropout(DROPOUT_PROB))
model.add(Dense(HIDDEN_LAYER_2_NODES, activation='relu'))
model.add(Dropout(DROPOUT_PROB))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

training_history = model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy)) 

model_structure = model.to_json()
with open("Output Files/sarcasm_detection_model_cnn.json", "w") as json_file:
    json_file.write(model_structure)
model.save_weights("Output Files/sarcasm_detection_model_cnn.h5")