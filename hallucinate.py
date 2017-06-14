#!/usr/bin/env python

import numpy
import os
import sys
from keras.models import Sequential

NUM_CHARS_HALLUCINATE = 1000

SEED_TEXT = "".join(sys.argv[1:])
if SEED_TEXT is "":
  SEED_TEXT = "there once was a girl who "
  print("No seed text presented on argv, using a default: " + SEED_TEXT)

MODELNAME = os.getenv('MODEL')
if MODELNAME is None:
  print("No MODEL= supplied! Aborting")
  os.exit(1)

input_dir = os.getenv("INPUT_DIR")
if input_dir is None:
  input_dir = "input"

# load input files so we can determine what "vocab" was used to train the model
# and we can scale our output
raw_input_files = glob.glob(input_dir + "/*.txt")
raw_text = ''
for f in raw_input_files:
  print("Loading input text: " + f)
  raw_text += open(f).read().lower()
chars = sorted(list(set(raw_text)))
n_vocab = len(chars)

# load the network weights
model = Sequential()
model.load_weights(MODELNAME)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# we need to translate characters back to human readable from ints (ord()),
# so use chr() when hallucinating.

# pick a random seed
pattern = [ord(c) for c in SEED_TEXT]
print("Seed text:")
print( "\"", ''.join([ord(value) for value in pattern]), "\"")
# generate characters
for i in range(NUM_CHARS_HALLUCINATE):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
  # normalize given the training set of data
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = chr(index)
	seq_in = [chr(value) for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")
