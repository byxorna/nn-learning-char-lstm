#!/usr/bin/env python
import os
import glob
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

input_dir = os.getenv("INPUT_DIR")
if input_dir is None:
  input_dir = "input"

# load ascii text and covert to lowercase
raw_input_files = glob.glob(input_dir + "/*.txt")
raw_text = ''
for f in raw_input_files:
  print("Loading input text: " + f)
  raw_text += open(f).read().lower()

print("Loaded " + str(len(raw_text)) + " total characters")

# TODO: clean up text to remove undesirable characters
# ['\n', ' ', '!', '(', ')', '*', ',', '-', '.', ':', ';', '?', '[', ']', '_',
# ‘', '’', '“', '”', '\ufeff']

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
#print(chars)

n_chars = len(raw_text)
n_vocab = len(chars)
print( "Total Characters: ", n_chars)
print( "Total Vocab: ", n_vocab)

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
  seq_in = raw_text[i:i+seq_length] # input vector is the 100char context preceding target
  seq_out = raw_text[i+seq_length] # target vector is the next char
  dataX.append(char_to_int[c] for c in seq_in)
  dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print( "Total patterns: " + str(n_patterns))
