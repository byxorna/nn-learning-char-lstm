#!/usr/bin/env python

import numpy
import os
from keras.models import Sequential

seed_text = "there once was a girl who "
filename = os.getenv('MODEL')
if filename is None:
  print("No MODEL= supplied! Aborting")
  os.exit(1)

# load the network weights
model = Sequential()
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# we need to translate characters back to human readable from ints (ord()),
# so use chr() when hallucinating.

# pick a random seed
pattern = [ord(c) for c in seed_text]
print("Seed text:")
print( "\"", ''.join([ord(value) for value in pattern]), "\"")
# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = chr(index)
	seq_in = [chr(value) for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")
