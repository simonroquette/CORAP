import random #Kind of paradoxal to import random in constants file ;)
import nltk.tokenize.casual


DEFAULT_N_UNITS = 650
DEFAULT_BATCHSIZE = 20
DEFAULT_CHECKPOINT = 4

# SHOULD REMOVE/ADD cap letter depending if model takes them in consideration
#Having capital letters means when you see a word, should also add it's capital version to the dictionnary !!
# ABCDEFGHIJKLMNOPQRSTUVWXYZ
alph = "abcdefghijklmnopqrstuvwxyz0123456789 .,:;'*!?`$%&(){}[]-/\@_#" #TODO Should space be part of our alphabet ??? probably if want deal spaced out words

# replacement_alph = "abcdefghijklmnopqrstuvwxyz0123456789"

ID_UNKNOWN_WORD = -1

def clean_input(s) :
    return ''.join([c for c in s.replace("\n", "<eos>").replace(" n ", " ").replace(" N ", " ") if c in alph])

def my_tokenize(s) :
    return nltk.tokenize.casual.casual_tokenize(clean_input(s), preserve_case = False)

MAX_WORD_LENGTH = 16
data_dim = len(alph)*MAX_WORD_LENGTH

MIN_OCCURENCES = 2 # The number of times a word must be seen in training to be considered as a vocabulary word

DROPOUT_RATE_INPUT = 0.01
DROPOUT_RATE = 0.5

MISTAKE_PROBABILITY = 0.3 # The probability that a word has a mistake in a text (what is good value for OCR ??)
def mistake_happen():
    return random.random() < MISTAKE_PROBABILITY
