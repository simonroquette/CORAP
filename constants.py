import random #Kind of paradoxal to import random in constants file ;)

# SHOULD REMOVE/ADD cap letter depending if model takes them in consideration
#Having capital letters means when you see a word, should also add it's capital version to the dictionnary !!
# ABCDEFGHIJKLMNOPQRSTUVWXYZ
alph = " abcdefghijklmnopqrstuvwxyz0123456789.,:;'*!?`$%&(){}[]-/\@_#" #TODO Should space be part of our alphabet ??? probably if want deal spaced out words


MAX_WORD_LENGTH = 16
data_dim = len(alph)*MAX_WORD_LENGTH

MIN_OCCURENCES = 1 # The number of times a word must be seen in training to be considered as a vocabulary word

DROPOUT_RATE_INPUT = 0.01
DROPOUT_RATE = 0.5

MISTAKE_PROBABILITY = 0.3 # The probability that a word has a mistake in a text (what is good value for OCR ??)
def mistake_happen():
    return random.random() < MISTAKE_PROBABILITY
