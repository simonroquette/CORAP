#!/usr/bin/env python
# encoding: utf-8

import string
import numpy as np
import random

from constants import *


# ADDED_NOTE comments by SIMON ROQUETTTE, REPLACETABLE with its frame too

# TODO insert space
# delete space
# add '-' ? (two words beginning/ending on a different line ?)
# Can't put 'i' as ressemblance to 1, because our model ignores words with a number for the moment...
# Faire varier probabilités chaque evenement et voir performances
# Search more equivalences

# EQUIVALENCE_TABLE and EQUIVALENCE_DICT should be build at the beginning, only one time !

# Don't put the same char in two equivalence tables. It will be overwritten.
# Indeed, equivalence is a transitive property, if A looks like B and B looks like C, A looks like C
# And therefore they should be in the same table

# I put each alpha character in equivalence table, maybe better to take some out ? TODO
EQUIVALENCE_TABLE = [['i', 'j', 'l'],
                     ['n', 'r'],
                     ['m', 'nn', 'rn', 'nr'],
                     ['mm', 'nnm', 'mnn', 'nnn', 'nnnn', 'rnm', 'nrm', 'mrn'],
                     ['u', 'v', ],
                     ['w', 'vv'],
                     ['o', 'a'],
                     #['s', 'z'],  # s and z depend on the hand writing...
                     #['x', 'ae', 'oe', 'oc'],  # Maybe too intense
                     ['g', 'q'],
                     #['k', 'le', 'lR'],
                     ['c', 'e'],
                     ['h', 'b']
                     #['d', 't']  # Maybe too intense
                     ]

# Build dictionnary once, because it is faster when looking for equivalences
EQUIVALENCE_DICT = dict()

for line in range(len(EQUIVALENCE_TABLE)):
    for entry in range(len(EQUIVALENCE_TABLE[line])):
        if len(EQUIVALENCE_TABLE[line][entry]) <= 2:  # We don't care entries that are bigger than 2
            copy = list(EQUIVALENCE_TABLE[line])
            del copy[entry]
            EQUIVALENCE_DICT[EQUIVALENCE_TABLE[line][entry]] = copy


# MANUALLY ADD SOME EQUIVALENCES HERE THAT ARE ONLY ONE WAY: EQUIVALENCES_DICT[to_add] = [add1, add2]


def hasnum(w):
    for c_i in w:
        if c_i.isdigit():
            return True
    return False


# "a" = 97
def shape_input(w) : # Not used
    num = []

    for c in w:
        if(isalnum(c)):
            if(c.isdigit()) :
                num.append( 28 * 1000)
            else :
                num.append( (ord(c.lower())-ord("a")) * 1000) #to space them out
        else :
            num.append(30 * 1000)

    return np.array(num)



def noise_char(w, opt, alph):

    bin = [0] * len(alph) * MAX_WORD_LENGTH

    if w == '<eos>':
        for i in range(MAX_WORD_LENGTH) :
            bin[(i+1)*len(alph) - 1] += 1
    elif w == '<unk>': # Should be removed ???
        #print("UNK here !!!!!!!!!!!!!!")
        for i in range(MAX_WORD_LENGTH) :
            bin[(i+1)*len(alph) - 2] += 1

    # Events with no noise here : - numbers
    # - ponctuation like "!" "??" or "),"
    elif hasnum(w) or ((not w.isalpha()) and len(w) < 3):
        for i in range(len(w)):
            bin[i*len(alph) + alph.index(w[i])] += 1

    else:
        if opt == "DELETE" and len(w) > 1: # Words of length 1 don't overgo deletion...
                idx = random.randint(0, len(w) - 1)
                w = w[:idx] + w[idx + 1:]

        if opt == "INSERT":
            ins_idx = random.randint(0, len(w) - 1)
            ins_char_idx = np.random.randint(0, len(string.ascii_lowercase))
            ins_char = list(string.ascii_lowercase)[ins_char_idx]
            w = w[:ins_idx] + ins_char + w[ins_idx:]

        if opt == "REPLACE":
            target_idx = random.randint(0, len(w) - 1)
            rep_char_idx = np.random.randint(0, len(string.ascii_lowercase))
            rep_char = list(string.ascii_lowercase)[rep_char_idx]
            w = w[:target_idx] + rep_char + w[target_idx + 1:]

        if opt == "REPLACETABLE":
            # Choisir un nombre aleatoire: index du char à changer
            # Les fusions de deux caracteres sont les plus probables, peut etre modifie...
            choices = list(range(len(w)))
            added = ''
            while len(choices) > 0 and added == '':
                idx = random.randint(0, len(choices)-1)
                i = choices[idx]
                if i == len(w) - 1:
                    if (i > 0) and w[i - 1] + w[i] in EQUIVALENCE_DICT:
                        added = random.choice(EQUIVALENCE_DICT[w[i - 1] + w[i]])
                        w = w[:i - 1] + added
                    elif w[i] in EQUIVALENCE_DICT:
                        added = random.choice(EQUIVALENCE_DICT[w[i]])
                        w = w[:i] + added
                    else:
                        del choices[idx]

                elif i == len(w) - 2:
                    if w[i] + w[i + 1] in EQUIVALENCE_DICT:
                        added = random.choice(EQUIVALENCE_DICT[w[i] + w[i + 1]])
                        w = w[:i] + added
                    elif (i > 0) and w[i - 1] + w[i] in EQUIVALENCE_DICT:
                        added = random.choice(EQUIVALENCE_DICT[w[i - 1] + w[i]])
                        w = w[:i - 1] + added + w[i + 1]
                    elif w[i] in EQUIVALENCE_DICT:
                        added = random.choice(EQUIVALENCE_DICT[w[i]])
                        w = w[:i] + added + w[i + 1:]
                    else:
                        del choices[idx]

                else:
                    if w[i] + w[i + 1] in EQUIVALENCE_DICT:
                        added = random.choice(EQUIVALENCE_DICT[w[i] + w[i + 1]])
                        w = w[:i] + added + w[i + 2:]
                    elif (i > 0) and w[i - 1] + w[i] in EQUIVALENCE_DICT:
                        added = random.choice(EQUIVALENCE_DICT[w[i - 1] + w[i]])
                        w = w[:i - 1] + added + w[i + 1:]
                    elif w[i] in EQUIVALENCE_DICT:
                        added = random.choice(EQUIVALENCE_DICT[w[i]])
                        w = w[:i] + added + w[i + 1:]
                    else:
                        del choices[idx]

        for i in range(min(len(w), MAX_WORD_LENGTH)): #Last letters of a word longer than MAX_WORD_LENGTH will be ignored
            #TODO One other way could be to ignore middle ones ? Because last letters matter more (gender/plural, is more important)
            bin[i*len(alph) + alph.index(w[i])] += 1

    return np.array(bin), w


def jumble_char(w, opt, alph): #Not adapted new version
    if opt == "WHOLE":
        bin_all = [0] * len(alph)
        bin_filler = [0] * (len(alph) * 2)
        if w == '<eos>':
            bin_all[-1] += 1
        elif w == '<unk>':
            bin_all[-2] += 1
        elif hasnum(w):
            bin_all[-3] += 1
        else:
            w = ''.join(random.sample(w, len(w)))
            for i in range(len(w)):
                try:
                    bin_all[alph.index(w[i])] += 1
                except ValueError:
                    print(w)
                    raise

        bin_all = bin_all + bin_filler
        return np.array(bin_all), w

    elif opt == "BEG":
        bin_initial = [0] * len(alph)
        bin_end = [0] * len(alph)
        bin_filler = [0] * len(alph)
        if w == '<eos>':
            bin_initial[-1] += 1
            bin_end[-1] += 1
        elif w == '<unk>':
            bin_initial[-2] += 1
            bin_end[-2] += 1
        elif hasnum(w):
            bin_initial[-3] += 1
            bin_end[-3] += 1
        else:
            if len(w) > 3:
                w_init = ''.join(random.sample(w[:-1], len(w[:-1])))
                w = w_init + w[-1]
            for i in range(len(w)):
                try:
                    if i == len(w) - 1:
                        bin_end[alph.index(w[i])] += 1
                    else:
                        bin_initial[alph.index(w[i])] += 1
                except ValueError:
                    print(w)
                    raise
        bin_all = bin_initial + bin_end + bin_filler
        return np.array(bin_all), w

    elif opt == "END":
        bin_initial = [0] * len(alph)
        bin_end = [0] * len(alph)
        bin_filler = [0] * len(alph)
        if w == '<eos>':
            bin_initial[-1] += 1
            bin_end[-1] += 1
        elif w == '<unk>':
            bin_initial[-2] += 1
            bin_end[-2] += 1
        elif hasnum(w):
            bin_initial[-3] += 1
            bin_end[-3] += 1
        else:
            if len(w) > 3:
                w_end = ''.join(random.sample(w[1:], len(w[1:])))
                w = w[0] + w_end
            for i in range(len(w)):
                try:
                    if i == 0:
                        bin_initial[alph.index(w[i])] += 1
                    else:
                        bin_end[alph.index(w[i])] += 1
                except ValueError:
                    print(w)
                    raise
        bin_all = bin_initial + bin_end + bin_filler
        return np.array(bin_all), w

    elif opt == "INT":
        bin_initial = [0] * len(alph)
        bin_middle = [0] * len(alph)
        bin_end = [0] * len(alph)
        if w == '<eos>':
            bin_initial[-1] += 1
            bin_middle[-1] += 1
            bin_end[-1] += 1
        elif w == '<unk>':
            bin_initial[-2] += 1
            bin_middle[-2] += 1
            bin_end[-2] += 1
        elif hasnum(w):
            bin_initial[-3] += 1
            bin_middle[-3] += 1
            bin_end[-3] += 1
        else:
            if len(w) > 3:
                w_mid = ''.join(random.sample(w[1:-1], len(w[1:-1])))
                w = w[0] + w_mid + w[-1]
            for i in range(len(w)):
                try:
                    if i == 0:
                        bin_initial[alph.index(w[i])] += 1
                    elif i == len(w) - 1:
                        bin_end[alph.index(w[i])] += 1
                    else:
                        bin_middle[alph.index(w[i])] += 1
                except ValueError:
                    print(w)
                    raise
        bin_all = bin_initial + bin_middle + bin_end
        return np.array(bin_all), w
    else:
        raise


if __name__ == "__main__":
    word = 'research'
    print(word)
    v, w = noise_char(word, 'DELETE', alph)
    print(w)
    v, w = noise_char(word, 'INSERT', alph)
    print(w)
    v, w = noise_char(word, 'REPLACE', alph)
    print(w)
