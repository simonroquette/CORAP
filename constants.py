import random #Kind of paradoxal to import random in constants file ;) 

alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;'*!?`$%&(){}[]-/\@_#" # SHOULD REMOVE/ADD cap letter depending if model takes them in consideration
#Having capital letters means when you see a word, should also add it's capital version to the dictionnary !!

MAX_WORD_LENGTH = 16
data_dim = len(alph)*MAX_WORD_LENGTH

DROPOUT_RATE_INPUT = 0.01
DROPOUT_RATE = 0.5

MISTAKE_PROBABILITY = 0.33 # The probability that a word has a mistake in a text (what is good value for OCR ??)
def mistake_happen():
    return random.random() < MISTAKE_PROBABILITY
