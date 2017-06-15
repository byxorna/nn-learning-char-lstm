#!/usr/bin/env python

from loader import TextLoader
import sys
import glob
import argparse
import numpy
import keras

parser = argparse.ArgumentParser(
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--chars', type=int, default=1000, help='Number of characters to hallucinate')
parser.add_argument('--input-dir', type=str, default="./input", help='Input dir to search for training data')
parser.add_argument('--model', type=str, help='Model to load')
#parser.add_argument('seed_text', type=str, default="there once was a girl who ", nargs="*")
args = parser.parse_args()
print(args)

if args.model is None:
  print("No model supplied! Aborting")
  sys.exit(1)


# load input files so we can determine what "vocab" was used to train the model
# and we can scale our output
loader = TextLoader()
loader.load(files=glob.glob(args.input_dir + "/*.txt"))

# load and compile model, including weights and training config
model = keras.models.load_model(args.model)
#model.load_weights(args.model)
#model.compile(loss='categorical_crossentropy', optimizer='adam')

# we need to translate characters back to human readable from ints (ord()),
# so use chr() when hallucinating.

# need to reshape input???
#pattern = [ord(c) for c in args.seed_text.lower()]
#print("Seed text: " + ''.join([chr(value) for value in pattern]))


# pick a random seed
# remember, our model is predicting characters given 100 char inputs, so we need to seed it with something
start = numpy.random.randint(0, len(loader.raw_text)-1)
pattern = [loader.char_to_int[c] for c in loader.raw_text[start:start+100]]
print("".join(loader.raw_text[start:start+100]))

# generate characters
for i in range(args.chars):
  x = numpy.reshape(pattern, (1, len(pattern), 1))
  # normalize given the training set of data, with the vocab of our seed phrase
  x = x / float(len(loader.uniq_chars))
  prediction = model.predict(x, verbose=0)
  index = numpy.argmax(prediction)
  result = chr(index)
  #seq_in = [chr(value) for value in pattern]
  sys.stdout.write(result)
  sys.stdout.flush()
  pattern.append(index)
  pattern = pattern[1:len(pattern)]
