# CORAP
Spelling corrector project based on Deep Learning

#How to use
To use predict.py, use option -m "PATH_TO_MODEL"

To use correct.py :
- Change MODEL_PATH in code to use another model
- option -f to give file to correct
- option -t to correct the given text in console

To use noise_generator.py :
- Results will be stored in /data/errors.txt from /data/source.txt by default, but path can be specified with option -f -s
- option -t to give a text in console
- option -c to see the noise in console only


#Model infos
For the moment model is set to ignore capital letters
