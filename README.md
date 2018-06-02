# CORAP (Correction ORthographique par Apprentissage Profond)

Spelling correction based on Deep Learning.

This is a project I did for Professor Michel Gagnon, under the supervision of Claude Coulombe, in Polytechnique Montreal. I started in January 2018 and stopped working on it in May 2018.

The aim of this project was to build a deep learning based program to correct spelling mistakes related to OCR noise (on handwritten texts). Claude Coulombe needed this for his own work, and I therefore had to build it as a usable tool, rather than simply showing interesting results and learning performances.

## Inspiration

As I didn't know anything about deep learning when I started, I decided to start on someone else's code, learn with it and then modify it to meet my specific requirements.

So here is a big thank you to the Johns Hopkins University team who wrote [this article](https://arxiv.org/pdf/1608.02214.pdf), and a special thank you to Keisuke Sakaguchi who took the time to answer my questions ! I took their code and simply added/modified what I needed all along.

The model is very closed to the one in the article, so take a look at it if you want a general idea ! The main difference is that my model is a CharRNN, rather than scRNN, meaning all characters are injected the same way in the LSTM units.

If I had to start again today, I would rather start from nothing, as some parts of the code are not useful in my project. I didn't do so because, again, I had no knowledge in deep learning at first.

# Contacts

 - simon.roquette@epfl.ch (student who wrote the code)
 - claude.coulombe@gmail.com (Supervisor)
 - michel.gagnon@polymtl.ca (Professor)

# How to use

 - _predict.py_ : To see performance on test data

 use option -m "PATH_TO_MODEL"

 - _correct.py_ : Corrects given file or text using a pre-trained model

Change MODEL_PATH in code to use another model
option -f to give file to correct
option -t to correct the given text in console

 - _noise_generator.py_ : Generates noised text from a source, to see

Results will be stored in /data/errors.txt from /data/source.txt by default, but path can be specified with option -f -s
option -t to give a text in console
option -c to see the noise in console only

# Data generation

Because no big OCR texts with spelling mistakes and their correction exist to my knowledge (if you happen to know any good one, please contact me !), we take a correctly spelled text, and add noise to it in order to have training data. I build an OCR noise fonction based on typical mistakes I read on internet, and hard to read letters I saw on my friends' handwriting. There are kind of mistakes randomly added :
 - (20%) Random deletion of a letter
 - (20%) Random addition of a letter
 - (20%) Random replacement of a letter by another (random one)
 - (40%) Replacement of a letter or sequence of letters by another one that looks alike (*nn* with *m*, *u* with *v*... see binarize.py EQUIVALENCE_TABLE for more details)

# Model

### Results

Because my code is not supposed to only see how well a deep learning model can correct mistakes, but to be a usable solution to spelling correction, there are therfore two kinds of mistakes
 - False positives (doesn't correct a word that has a spelling mistake, happens often when for example *an* is mapped to *a*)
 - True negative (corrects a word that didn't contain a spelling mistake)
It has an overall 94% accuracy, on my noise considering 1/3 word has a spelling mistake.

### What it is
- *CharRNN*
- LSTM units stacked (didn't change it from the code I based my work on, except input shape)
- Classification problem : dictionnary is fed with training data's words. Each word seen is mapped to the most probable output in the build dictionnary

### To do/improve
- (improve) There is a probability that the word is unknown and it will be mapped to himself (useful for unknown first names etc...) To train the model to this, words that are only seen once or twice are withdrawn from the dictionnary
- For the moment model is set to ignore capital letters
- Much more ;)

# About me

My name is Simon Roquette, I am a french EPFL (Ecole Polytechnique Fédérale de Lausanne) student, who went on an exchange in Polytechnique Montreal where I made this work. My field of study is Communication Systems, and I recently specialized in Data Science. I only discovered natural language processing, artificial intelligence and deep learning through this project. This can explain why this it is not perfect at all, and I strongly encourage you to contact me if you have any question regarding my approach, my code or anything. Thank you for your consideration !
