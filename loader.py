#!/usr/bin/env python
import re
import os
import numpy

class TextLoader:
  # Class TextLoader will load a set of training data from a directory,
  # and provide you with data about it, like number of patterns, char count,
  # etc.
  # TODO: clean up text to remove undesirable characters
  #UNDESIRABLE_CHARS = ['\n', '!', '(', ')', '*', '-', ':', ';', '[', ']', '_', '‘', '’', '“', '”', '\ufeff']
  def __init__(self):
    # load ascii text and covert to lowercase
    self.raw_text = ""
    self.num_chars = 0
    self.uniq_chars = []
    self.num_words = 0
    self.uniq_words = []

  def load(self, files):
    """given a list of files, read their contents. NOTE: this resets
    internal state, so if you call load twice, it will clobber the first
    loaded contents."""
    self.raw_text = ""
    for f in files:
      #print("Loading input text from " + f)
      loaded_text = open(f).read().lower()
      self.raw_text += loaded_text
      #''.join([c for c in loaded_text if c not in self.UNDESIRABLE_CHARS])
    #"""the length of the raw text string"""
    self.num_chars = len(self.raw_text)
    #"""the unique sorted set of all characters in the loaded text"""
    self.uniq_chars = sorted(list(set(self.raw_text)))
    #"""returns a dictionary of char to int representation
    #of the current loaded training data set"""
    self.char_to_int = dict((c, i) for i, c in enumerate(self.uniq_chars))

    # get rid of garbage, this is good enough. dont let any punctuation in
    self.words = re.split('[^a-zA-Z\']+', self.raw_text)
    self.num_words = len(self.words)
    self.uniq_words = sorted(list(set(self.words)))
    self.word_to_int = dict((w,i) for i, w in enumerate(self.uniq_words))

