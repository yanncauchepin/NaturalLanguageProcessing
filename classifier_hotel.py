import os
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
from keras.utils import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding

root_path = "/media/yanncauchepin/ExternalDisk/Datasets/NaturalLanguageProcessing/classifier_hotel/"
input_path = "data/makemytrip_com_travel_sample.csv"


dataset_path = os.path.join(root_path, input_path)

data = pd.read_csv(dataset_path, index_col=0)

print(data.head())

print(data.city.value_counts())

data = data.loc[data['city'].isin(['Mumbai'])]

print(data.head())

data = data.hotel_overview
data = data.dropna()

stop = set(stopwords.words('english'))
def stopwords_removal(data_point):
    data = [x for x in data_point.split() if x not in stop]
    return data

def clean_data(data):
    cleaned_data = []
    all_unique_words_in_each_description = []
    for entry in data:
        entry = re.sub(pattern='[^a-zA-Z]',repl=' ',string = entry)
        entry = re.sub(r'\b\w{0,1}\b', repl=' ',string = entry)
        entry = entry.lower()
        entry = stopwords_removal(entry)
        cleaned_data.append(entry)
        unique = list(set(entry))
        all_unique_words_in_each_description.extend(unique)
    return cleaned_data, all_unique_words_in_each_description

def unique_words(data):
    unique_words = set(all_unique_words_in_each_description)
    return unique_words, len(unique_words)

cleaned_data, all_unique_words_in_each_description = \
    clean_data(data)
unique_words, length_of_unique_words = \
    unique_words(all_unique_words_in_each_description)
    
"""
The cleaned_data parameter contains our preprocessed data.
The unique_words parameter contains our list of unique words.
The length_of_unique_words parameter is the number of unique words in the data.
"""

print(cleaned_data[0])
print(length_of_unique_words)

def build_indices(unique_words):
    word_to_idx = {}
    idx_to_word = {}
    for i, word in enumerate(unique_words):
        word_to_idx[word] = i
        idx_to_word[i] = word
    return word_to_idx, idx_to_word

word_to_idx, idx_to_word = build_indices(unique_words)

def prepare_corpus(corpus, word_to_idx):
    sequences = []
    for line in corpus:
        tokens = line
        for i in range(1, len(tokens)):
            i_gram_sequence = tokens[:i+1]
            i_gram_sequence_ids = []
            for j, token in enumerate(i_gram_sequence):
                i_gram_sequence_ids.append(word_to_idx[token])
            sequences.append(i_gram_sequence_ids)
    return sequences

sequences = prepare_corpus(cleaned_data, word_to_idx)
max_sequence_len = max([len(x) for x in sequences])

"""
The sequences parameter contains all the sequences from our data.
The max_sequence_len parameter conveys the length of the maximum sequence size that was built based on our data.
"""

print(sequences[0])
print(sequences[1])

print(idx_to_word[1647])
print(idx_to_word[867])
print(idx_to_word[1452])

print(len(sequences))
print(max_sequence_len)

def build_input_data(sequences, max_sequence_len, \
                     length_of_unique_words):
    sequences = np.array(pad_sequences(sequences, \
                    maxlen = max_sequence_len, padding = 'pre'))
    X = sequences[:,:-1]
    y = sequences[:,-1]
    y = np_utils.to_categorical(y, length_of_unique_words)
    return X, y

X, y = build_input_data(sequences, max_sequence_len, length_of_unique_words)

def create_model(max_sequence_len, length_of_unique_words):
    model = Sequential()
    model.add(Embedding(length_of_unique_words, 10, \
                        input_length=max_sequence_len - 1))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(length_of_unique_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', \
                  optimizer='adam')
    return model

model = create_model(max_sequence_len, length_of_unique_words)
model.summary()

model.fit(X, y, batch_size = 512, epochs=100)

def generate_text(seed_text, next_words, model, max_seq_len):
    for _ in range(next_words):
        cleaned_data = clean_data([seed_text])
        sequences= prepare_corpus(cleaned_data[0], word_to_idx)
        sequences = pad_sequences([sequences[-1]], maxlen=max_seq_len-1, \
                                  padding='pre')
        predicted = model.predict_classes(sequences, verbose=0)
        output_word = ''
        output_word = idx_to_word[predicted[0]]
        seed_text = seed_text + " " + output_word
    return seed_text.title()

print(generate_text("in Mumbai there we need", 30, model, max_sequence_len))

print(generate_text("The beauty of the city", 30, model, max_sequence_len))