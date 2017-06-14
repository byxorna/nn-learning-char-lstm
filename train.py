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

input_dir = os.environ["INPUT_DIR"]
if input_dir is None:
  input_dir = "input"

# load ascii text and covert to lowercase
raw_input_files = glob.glob(input_dir + "/*.txt")
raw_text = ''
for f in raw_input_files:
  raw_text += open(f).read().lower()

print("Read in characters ")
print(len(raw_text))
