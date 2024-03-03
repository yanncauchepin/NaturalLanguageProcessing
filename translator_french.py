import os
import pandas as pd
import string
import re
import io
import numpy as np
from unicodedata import normalize
import keras, tensorflow
from keras.models import Model
from keras.layers import Input, LSTM, Dense

def read_data(file):
    data = []
    with io.open(file, 'r') as file:
        for entry in file:
            entry = entry.strip()
            data.append(entry)
    return data

root_path = "/media/yanncauchepin/ExternalDisk/Datasets/NaturalLanguageProcessing/translator_french/"
input_path = "fra_eng/fra.txt"


dataset_path = os.path.join(root_path, input_path)


data = read_data(dataset_path)

print(data[139990:140000])

print(len(data))

def build_english_french_sentences(data):
    english_sentences = []
    french_sentences = []
    for data_point in data:
        english_sentences.append(data_point.split("\t")[0])
        french_sentences.append(data_point.split("\t")[1])
    return english_sentences, french_sentences

english_sentences, french_sentences = build_english_french_sentences(data)

def clean_sentences(sentence):
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    cleaned_sent = normalize('NFD', sentence).encode('ascii', \
                                                     'ignore')
    cleaned_sent = cleaned_sent.decode('UTF-8')
    cleaned_sent = cleaned_sent.split()
    cleaned_sent = [word.lower() for word in cleaned_sent]
    cleaned_sent = [word.translate(table) for word in cleaned_sent]
    cleaned_sent = [re_print.sub('', w) for w in cleaned_sent]
    cleaned_sent = [word for word in cleaned_sent if \
                    word.isalpha()]
    return ' '.join(cleaned_sent)

def build_clean_english_french_sentences(english_sentences, french_sentences):
    french_sentences_cleaned = []
    english_sentences_cleaned = []
    for sent in french_sentences:
        french_sentences_cleaned.append(clean_sentences(sent))
    for sent in english_sentences:
        english_sentences_cleaned.append(clean_sentences(sent))
    return english_sentences_cleaned, french_sentences_cleaned

english_sentences_cleaned, french_sentences_cleaned = build_clean_english_french_sentences(english_sentences, french_sentences)

def build_data(english_sentences_cleaned, french_sentences_cleaned):
    input_dataset = []
    target_dataset = []
    input_characters = set()
    target_characters = set()
    
    for french_sentence in french_sentences_cleaned:
        input_datapoint = french_sentence
        input_dataset.append(input_datapoint)
        for char in input_datapoint:
            input_characters.add(char)
        
    for english_sentence in english_sentences_cleaned:
        target_datapoint = "\t" + english_sentence + "\n"
        target_dataset.append(target_datapoint)
        for char in target_datapoint:
            target_characters.add(char)
            
    return input_dataset, target_dataset, \
           sorted(list(input_characters)), \
           sorted(list(target_characters)) 

input_dataset, target_dataset, input_characters, target_characters = build_data(english_sentences_cleaned, french_sentences_cleaned)

print(input_characters)

print(target_characters)


def build_metadata(input_dataset, target_dataset, \
                   input_characters, target_characters):    
    num_Encoder_tokens = len(input_characters)
    num_Decoder_tokens = len(target_characters)
    max_Encoder_seq_length = max([len(data_point) for data_point \
                                  in input_dataset]) 
    max_Decoder_seq_length = max([len(data_point) for data_point \
                                  in target_dataset])
    print('Number of data points:', len(input_dataset))
    print('Number of unique input tokens:', num_Encoder_tokens)
    print('Number of unique output tokens:', num_Decoder_tokens)
    print('Maximum sequence length for inputs:', \
           max_Encoder_seq_length)
    print('Maximum sequence length for outputs:', \
           max_Decoder_seq_length)
    return num_Encoder_tokens, num_Decoder_tokens, \
           max_Encoder_seq_length, max_Decoder_seq_length

num_Encoder_tokens, num_Decoder_tokens, max_Encoder_seq_length, max_Decoder_seq_length = build_metadata(input_dataset, target_dataset, input_characters, target_characters)

def build_indices(input_characters, target_characters):
    input_char_to_idx = {}
    input_idx_to_char = {}
    target_char_to_idx = {}
    target_idx_to_char = {}
    
    for i, char in enumerate(input_characters):
        input_char_to_idx[char] = i
        input_idx_to_char[i] = char
    
    for i, char in enumerate(target_characters):
        target_char_to_idx[char] = i
        target_idx_to_char[i] = char
    
    return input_char_to_idx, input_idx_to_char, \
           target_char_to_idx, target_idx_to_char

input_char_to_idx, input_idx_to_char, target_char_to_idx, target_idx_to_char = build_indices(input_characters, target_characters)

def build_data_structures(length_input_dataset, max_Encoder_seq_length, max_Decoder_seq_length, num_Encoder_tokens, num_Decoder_tokens):
    
    Encoder_input_data = np.zeros((length_input_dataset, \
      max_Encoder_seq_length, num_Encoder_tokens), dtype='float32')
    Decoder_input_data = np.zeros((length_input_dataset, \
      max_Decoder_seq_length, num_Decoder_tokens), dtype='float32')
    Decoder_target_data = np.zeros((length_input_dataset, \
      max_Decoder_seq_length, num_Decoder_tokens), dtype='float32')
    print("Dimensionality of Encoder input data is : ", \
           Encoder_input_data.shape)
    print("Dimensionality of Decoder input data is : ", \
           Decoder_input_data.shape)
    print("Dimensionality of Decoder target data is : ", \
           Decoder_target_data.shape)
    return Encoder_input_data, Decoder_input_data, \
           Decoder_target_data

Encoder_input_data, Decoder_input_data, Decoder_target_data = build_data_structures(len(input_dataset), max_Encoder_seq_length, max_Decoder_seq_length, num_Encoder_tokens, num_Decoder_tokens)

def add_data_to_data_structures(input_dataset, target_dataset, Encoder_input_data, Decoder_input_data, Decoder_target_data):
    for i, (input_data_point, target_data_point) in \
            enumerate(zip(input_dataset, target_dataset)):
        for t, char in enumerate(input_data_point):
            Encoder_input_data[i, t, input_char_to_idx[char]] = 1.
        for t, char in enumerate(target_data_point):
            # Decoder_target_data is ahead of Decoder_input_data by 
            # one timestep
            Decoder_input_data[i, t, target_char_to_idx[char]] = 1.
            if t > 0:
                # Decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                Decoder_target_data[i, t - 1, \
                                    target_char_to_idx[char]] = 1.
    return Encoder_input_data, Decoder_input_data, \
           Decoder_target_data

Encoder_input_data, Decoder_input_data, Decoder_target_data = add_data_to_data_structures(input_dataset, target_dataset, Encoder_input_data, Decoder_input_data, Decoder_target_data)

batch_size = 256
epochs = 100
latent_dim = 256

Encoder_inputs = Input(shape=(None, num_Encoder_tokens))
Encoder = LSTM(latent_dim, return_state=True)
Encoder_outputs, state_h, state_c = Encoder(Encoder_inputs)
Encoder_states = [state_h, state_c]

Decoder_inputs = Input(shape=(None, num_Decoder_tokens))
Decoder_lstm = LSTM(latent_dim, return_sequences=True, \
                    return_state=True)
Decoder_outputs, _, _ = Decoder_lstm(Decoder_inputs, \
                        initial_state=Encoder_states)
Decoder_dense = Dense(num_Decoder_tokens, activation='softmax')
Decoder_outputs = Decoder_dense(Decoder_outputs)

model = Model(inputs=[Encoder_inputs, Decoder_inputs], outputs=Decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.summary()

model.fit([Encoder_input_data, Decoder_input_data], 
          Decoder_target_data, 
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)


model.save('Output Files/neural_machine_translation_french_to_english.h5')

Encoder_model = Model(Encoder_inputs, Encoder_states)

Decoder_state_input_c = Input(shape=(latent_dim,))
Decoder_state_input_h = Input(shape=(latent_dim,))
Decoder_states_inputs = [Decoder_state_input_h, \
                         Decoder_state_input_c]

Decoder_outputs, state_h, state_c = Decoder_lstm(Decoder_inputs, \
                                initial_state=Decoder_states_inputs)
Decoder_states = [state_h, state_c]
Decoder_outputs = Decoder_dense(Decoder_outputs)

Decoder_model = Model([Decoder_inputs] + Decoder_states_inputs,
                      [Decoder_outputs] + Decoder_states)

def decode_sequence(input_seq):

    states_value = Encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_Decoder_tokens))
    target_seq[0, 0, target_char_to_idx['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = Decoder_model.predict([target_seq]+ \
                                                     states_value)
    
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = target_idx_to_char[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or len(decoded_sentence) > \
            max_Decoder_seq_length):
              stop_condition = True
      

        target_seq = np.zeros((1, 1, num_Decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
    
        states_value = [h, c]
    
    return decoded_sentence


def decode(seq_index):
    input_seq = Encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_dataset[seq_index])
    print('Decoded sentence:', decoded_sentence)


decode(55000)
