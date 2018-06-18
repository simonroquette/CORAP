import binarize
from constants import *

import argparse
import pickle
import h5py
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, LSTM, TimeDistributed
from keras.optimizers import SGD

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', default="data/errors.txt",
    help='Path of file to correct, default errors.txt')
parser.add_argument('--text', '-t', default="",
    help='Use this option to correct the specified text')
parser.add_argument('--mute', '-m', default=False, action='store_true',
    help='If True, results will not be printed in the console (Default = False)')
parser.add_argument('--batchsize', '-b', type=int, default=20,
    help='learning minibatch size')

args = parser.parse_args()
PATH_FILE = args.file #default is "errors.txt"
MUTE = args.mute
batchsize = args.batchsize
text = args.text

PATH_CORRECTION = "correction.txt"

PATH_MODEL = "models/last_model.h5"
PATH_VOCAB = "models/vocab"
PATH_ID2VOCAB = "models/id2vocab"

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

vocab = load_obj(PATH_VOCAB)
id2vocab = load_obj(PATH_ID2VOCAB)

text_cleaned = ""

if text == "":
    text_cleaned = open(PATH_FILE).read()
else :
    text_cleaned = text
text_cleaned = my_tokenize(text_cleaned)

def vectorize_data(vec_cleaned, data_name): # training, dev, or test
    X_vec = np.zeros((int(len(vec_cleaned)/batchsize), batchsize, data_dim), dtype=np.bool)
    X_token = []
    # easy minibatch
    # https://docs.python.org/2.7/library/functions.html?highlight=zip#zip
    for m, mini_batch_tokens in enumerate(zip(*[iter(vec_cleaned)]*batchsize)):
        X_token_m = []
        x_mini_batch = np.zeros((batchsize, data_dim), dtype=np.bool)
        for j, token in enumerate(mini_batch_tokens):
            x_mini_batch[j], x_token = binarize.noise_char(token, "No noise", alph)
            X_token_m.append(x_token)

        X_vec[m] = x_mini_batch
        X_token.append(X_token_m)

        #percentage = int(m*100. / (len(vec_cleaned)/batchsize))
        #sys.stdout.write("\r%d %% %s" % (percentage, data_name))
        #print(str(percentage) + '%'),
        #sys.stdout.flush()
    print()
    return X_vec, X_token

X_test, X_src = vectorize_data(text_cleaned, 'for correction')

model = load_model(PATH_MODEL)

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

correction = open(PATH_CORRECTION, 'w')

for j in range(len(X_test)):
    x_raw = X_test[np.array([j])]
    src_j = " ".join(X_src[j])
    preds = model.predict_classes(x_raw, verbose=0)
    pred_j = decode_word(preds[0], X_src[j], calc_argmax=False)

    txt_pred = " ".join(pred_j)

    if not MUTE:
        print("src : ", src_j)
        print("pred : ", txt_pred)
    if text == "" :
        correction.write(txt_pred)

correction.close()
