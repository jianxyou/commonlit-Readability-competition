import re
import os
import numpy as np
import pandas as pd
import random
import math
import tensorflow as tf
import logging
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
from transformers import RobertaTokenizer, TFRobertaModel

# Configurations
# Number of folds for training
FOLDS = 5
# Max length
MAX_LEN = 250
# Get the trained model we want to use
MODEL = '../input/tf-robertaa'
# Let's load our model tokenizer
tokenizer = RobertaTokenizer.from_pretrained(MODEL)


# This function tokenize the text according to a transformers model tokenizer
def regular_encode(texts, tokenizer, maxlen = MAX_LEN):
    enc_di = tokenizer.batch_encode_plus(
        texts,
        padding = 'max_length',
        truncation = True,
        max_length = maxlen,
    )
    
    return np.array(enc_di['input_ids'])

# This function encode our training sentences
def encode_texts(x_test, MAX_LEN = 350):
    x_test = regular_encode(x_test.tolist(), tokenizer, maxlen = MAX_LEN)
    return x_test

# Function to build our model
def build_roberta_base_model(max_len = MAX_LEN):
    transformer = TFRobertaModel.from_pretrained(MODEL)
    input_word_ids = tf.keras.layers.Input(shape = (max_len, ), dtype = tf.int32, name = 'input_word_ids')
    sequence_output = transformer(input_word_ids)[0]
    # We only need the cls_token, resulting in a 2d array
    cls_token = sequence_output[:, 0, :]
    output = tf.keras.layers.Dense(1, activation = 'linear', dtype = 'float32')(cls_token)
    model = tf.keras.models.Model(inputs = [input_word_ids], outputs = [output])
    return model

# Function for inference
def roberta_base_inference():
    #predictions1 = distilroberta_base_inference()
    # Read our test data
    df = pd.read_csv('../input/commonlitreadabilityprize/test.csv')
    # Get text features
    x_test = df['excerpt']
    # Encode our text with Roberta tokenizer
    x_test = encode_texts(x_test, MAX_LEN)
    # Initiate an empty vector to store prediction
    predictions = np.zeros(len(df))
    # Predict with the 5 models (5 folds training)
    for i in range(FOLDS):
        print('\n')
        print('-'*50)
        print(f'Predicting with model {i + 1}')
        # Build model
        model = build_roberta_base_model(max_len = MAX_LEN)
        # Load pretrained weights
        model.load_weights(f'../input/roberta-pretrained/Roberta_Base_123_{i + 1}.h5')
        # Predict
        fold_predictions = model.predict(x_test).reshape(-1)
        # Add fold prediction to the global predictions
        predictions += fold_predictions / FOLDS
    # Save submissions
    
    
    df['target'] = predictions
    #df[['id', 'target']].to_csv('submission.csv', index = False)
    return df

ragdf = roberta_base_inference()
