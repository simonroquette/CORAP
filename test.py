import nltk.tokenize.casual

PATH_TRAIN = './data/movie_lines_cleaned_v2.txt'

words = nltk.tokenize.casual.casual_tokenize(open(PATH_TRAIN, "r").read(), preserve_case = False)
word_counter = dict()

for w in words:
    if w in word_counter:
        if w == "\n":
            word_counter["<eos>"] += 1
        else :
            word_counter[w] += 1
    else:
        if w == "\n":
            word_counter["<eos>"] = 1
        else :
            word_counter[w] = 1
            
print(word_counter)
