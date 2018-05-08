import argparse
import binarize
from constants import *

import random

parser = argparse.ArgumentParser()

parser.add_argument('--source', '-s', default="./data/source.txt",
    help='Path of file to put noise on, default /data/source.txt')
parser.add_argument('--output', '-o', default="./data/errors.txt",
    help='Path of file to put noise on, default /data/errors.txt')
parser.add_argument('--text', '-t', default="",
    help='A text to put noise on')
parser.add_argument('--console', '-c', default=False, action='store_true',
    help='If True, results will be printed in the console only')


args = parser.parse_args()

SOURCE_PATH = args.source
OUTPUT_FILE = open(args.output, "w")
SOURCE_TEXT = args.text
PRINT_CONSOLE = args.console

if SOURCE_TEXT == "":
    SOURCE_TEXT = open(SOURCE_PATH, "r").read()

OUTPUT_WORDS = [""]

text_cleaned = my_tokenize(SOURCE_TEXT)

print("\n")

for w in text_cleaned:
    if True: # mistake_happen() if want to make errors not all the time
        rnd_noise = random.choice(['DELETE', 'INSERT', 'REPLACE', 'REPLACETABLE', 'REPLACETABLE']) #MAKE REPLACETABLE MORE PROBABLE
        token, w = binarize.noise_char(w, rnd_noise, alph)

    if PRINT_CONSOLE :
        print(w + " ", end = "", flush=True)
    else :
        OUTPUT_FILE.write(w + " ")
