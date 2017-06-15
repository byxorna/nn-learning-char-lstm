#!/usr/bin/env python

from loader import TextLoader
import sys
import glob
import argparse
import numpy
import keras

parser = argparse.ArgumentParser(
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--words', type=int, default=100, help='Number of words to hallucinate')
parser.add_argument('--input-dir', type=str, default="./input", help='Input dir to search for training data')
parser.add_argument('--model', type=str, help='Model to load')
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
seq_length = 10
model = keras.models.load_model(args.model)
start = numpy.random.randint(0, len(loader.words)-seq_length)
pattern = [loader.word_to_int[c] for c in loader.words[start:start+seq_length]]
print(" ".join(loader.words[start:start+seq_length]))

def sample_prediction(prediction):
  rnd_idx = numpy.random.choice(len(prediction), p=prediction)
  return rnd_idx

# generate characters
for i in range(args.words):
  x = numpy.reshape(pattern, (1, len(pattern), 1))
  print("new pattern:",[loader.uniq_words[w] for w in pattern])
  # normalize given the training set of data, with the vocab of our seed phrase
  x = x / float(len(loader.uniq_words))
  prediction = model.predict(x, verbose=0)
  index = numpy.argmax(prediction)
  # TODO: add randomness into sampling the predictions?
  #index = sample_prediction(prediction)
  result = loader.uniq_words[index] # will this blow up if it produces a number we havent seen?
  sys.stdout.write(" " + result)
  sys.stdout.flush()
  pattern.append(index)
  pattern = pattern[1:len(pattern)]
