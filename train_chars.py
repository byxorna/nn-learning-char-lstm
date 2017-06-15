#!/usr/bin/env python
import os
import argparse
import glob
import numpy
from loader import TextLoader
import keras.models
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

parser = argparse.ArgumentParser(
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input-dir', type=str, default="./input", help='Input dir to search for training data')
parser.add_argument('--checkpoint-dir', type=str, default="./checkpoints", help='Dir to save checkpoint models to')
parser.add_argument('--model', type=str, default=None, help='Model to bootstrap training from')
args = parser.parse_args()

loader = TextLoader()
loader.load(files=glob.glob(args.input_dir + "/*.txt"))

n_chars = loader.num_chars
n_vocab = len(loader.uniq_chars)
print( "Total Characters in training set: ", n_chars)
print( "Vocabulary: ", n_vocab, loader.uniq_chars)

seq_length = 100
dataX = [] # will be an array of n_chars length, of char[100] patterns
dataY = []
for i in range(0, n_chars - seq_length, 1):
  seq_in = loader.raw_text[i:i+seq_length] # input vector is the 100char context preceding target
  seq_out = loader.raw_text[i+seq_length] # target vector is the next char
  dataX.append([loader.char_to_int[c] for c in seq_in])
  dataY.append(loader.char_to_int[seq_out])

n_patterns = len(dataX)

print( "Total patterns: " + str(n_patterns))

# reshape X to be [samples, time steps, features]
X = numpy.reshape(numpy.array(dataX), (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
if args.model is not None:
  model = keras.models.load_model(args.model)
else:
  lstm_units = 64 #256
  model = keras.models.Sequential()
  model.add(LSTM(lstm_units, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(lstm_units))
  model.add(Dropout(0.2))
  model.add(Dense(y.shape[1], activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint, make sure to write the full model out
filepath=args.checkpoint_dir + "/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, save_weights_only=False, monitor='loss', verbose=1, save_best_only=True, mode='min')

# now iterate on our model, and find the best model
print("Fitting model...")
model.fit(X, y, epochs=20, batch_size=64, callbacks=[checkpoint])
