#!/usr/bin/env python
from __future__ import print_function # compatible print function for py2 print()
import argparse
import math
import os
import sys
import time
import datetime
import random
import numpy as np
import six
import csv
import pickle

from constants import *
import binarize

import h5py
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, LSTM, TimeDistributed
from keras.optimizers import SGD

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', '-b', type=int, default=20,
    help='learning minibatch size')
parser.add_argument('--noise', '-n', default="OCR",
    help='noise type (JUMBLE, INSERT, DELETE, REPLACE, RANDOM, OCR)')
parser.add_argument('--jumble', '-j', default="INT",
    help='jumble position (INT, WHOLE, BEG, or END)')

PATH_TEST = './data/SHORT.ptb.test.txt'

PATH_VOCAB = "models/vocab"
PATH_ID2VOCAB = "models/id2vocab"
model_file = "models/model_94%.h5"

args = parser.parse_args()
batchsize = args.batchsize  # minibatch size
noise_type = args.noise     # noise type
jumble_type = args.jumble   # jumble position
assert noise_type in ['JUMBLE', 'INSERT', 'DELETE', 'REPLACE', 'RANDOM', 'OCR']
assert jumble_type in ['INT', 'WHOLE', 'BEG', 'END']
assert os.path.exists(model_file)
if not noise_type in ['JUMBLE', 'RANDOM']:
    jumble_type = "NO"

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

print("===== LOADING VOCAB =====")
vocab = load_obj(PATH_VOCAB)
id2vocab = load_obj(PATH_ID2VOCAB)

def colors(token, color='green'):
   c_green = '\033[92m' # green
   c_red = '\033[91m' # red
   c_close = '\033[0m' # close
   return c_green + token + c_close

def decode_word(X, src, calc_argmax):
    if calc_argmax:
        X = X.argmax(axis=-1)
    result = []
    for i in range(len(X)):
        if X[i] == ID_UNKNOWN_WORD:
            result.append(src[i])
        else:
            result.append(id2vocab[X[i]])
    return result


# sentence is represented as id, <eos> is also represented as one word

test_cleaned = my_tokenize(open(PATH_TEST).read())

print('#vocab:\t', len(vocab)-2) # excluding BOS, EOS
print('#tokens in test:\t', len(test_cleaned))


print("===== VECTORIZING DATA =====")
timesteps = len(test_cleaned)
data_dim = len(alph)*MAX_WORD_LENGTH

def vectorize_data(vec_cleaned, data_name): # training, dev, or test
    X_vec = np.zeros((int(len(vec_cleaned)/batchsize), batchsize, data_dim), dtype=np.bool)
    Y_vec = np.zeros((int(len(vec_cleaned)/batchsize), batchsize, len(vocab)), dtype=np.bool)
    X_token = []
    # easy minibatch
    # https://docs.python.org/2.7/library/functions.html?highlight=zip#zip
    for m, mini_batch_tokens in enumerate(zip(*[iter(vec_cleaned)]*batchsize)):
        X_token_m = []
        x_mini_batch = np.zeros((batchsize, data_dim), dtype=np.bool)
        y_mini_batch = np.zeros((batchsize, len(vocab)), dtype=np.bool)

        for j, token in enumerate(mini_batch_tokens):
            if not mistake_happen():
                x_mini_batch[j], x_token = binarize.noise_char(token, "NO NOISE", alph)
            elif noise_type == 'OCR':
                rnd_noise = random.choice(['DELETE', 'INSERT', 'REPLACE', 'REPLACETABLE', 'REPLACETABLE'])  # MAKE REPLACETABLE MORE PROBABLE
                x_mini_batch[j], x_token = binarize.noise_char(token, rnd_noise, alph)
            elif jumble_type == 'NO':
                x_mini_batch[j], x_token = binarize.noise_char(token, noise_type, alph)
                pass
            else:
                x_mini_batch[j], x_token = binarize.jumble_char(token, jumble_type, alph)

            bin_label = [0]*len(vocab)


            if token in vocab.keys():
                bin_label[vocab[token]] = 1
            else:
                bin_label[ID_UNKNOWN_WORD] = 1

            y_mini_batch[j] = np.array(bin_label)
            X_token_m.append(x_token)
        X_vec[m] = x_mini_batch
        Y_vec[m] = y_mini_batch
        X_token.append(X_token_m)

        #percentage = int(m*100. / (len(vec_cleaned)/batchsize))
        #sys.stdout.write("\r%d %% %s" % (percentage, data_name))
        #print(str(percentage) + '%'),
        #sys.stdout.flush()
    print()
    return X_vec, Y_vec, X_token


X_test, Y_test, X_test_token = vectorize_data(test_cleaned, 'for test data')

print("data shape (#_batches, batch_size, vector_size)")
print("X_test", X_test.shape)
print("Y_test", Y_test.shape)


#LOAD the model
model = load_model(model_file)

mots = 0
corrects = 0

for j in range(len(X_test)):

    x_raw, y_raw = X_test[np.array([j])], Y_test[np.array([j])]
    src_j = " ".join(X_test_token[j])
    ref_j = decode_word(y_raw[0], X_test_token[j], calc_argmax=True)
    preds = model.predict_classes(x_raw, verbose=0)
    pred_j = decode_word(preds[0], X_test_token[j], calc_argmax=False)
    # coloring
    #pred_j_list = pred_j.split()
    #ref_j_list = ref_j.split()
    for k in range(len(pred_j)):
        mots += 1
        if pred_j[k] == ref_j[k]:
            corrects += 1
            pred_j[k] = colors(pred_j[k])

    if j%10 == 0:
        print('example #', str(j+1))
        print('src: ', src_j)
        print('prd: ', " ".join(pred_j))
        print('ref: ', " ".join(ref_j))

    print("ref : ", y_raw[0])
    print("pred : ", preds[0])

print('Score en %: ', str(corrects/mots*100.0))
