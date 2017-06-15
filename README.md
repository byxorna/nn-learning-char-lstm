http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

NOTE: hallucination doesnt strictly speaking work yet. As of now, I have only gotten the network to spit out a trail of `"` or `the`. Need to dig in further...

# Training

`--input-dir` is used to read input text files in that directory to train on. No validation datasets are used; we train on everything to have the most data for character occurrence. Use `--model` to pick up where you left off training.

```
source bin/activate
pip install -r requirements.txt
./train.py --model ./checkpoints/sometempsave.hdf5
```

# Hallucinating

Pass a string to seed its hallucination, and an optional `--chars` flag (default 1000) to
generate some number of characters. `--input-dir` defaults to `./input` to read in the training
data sets, so we can normalize the pattern vectors when loading the model. `--model` is used
to parameterize the model with a trained network.

```
source bin/activate
pip install -r requirements.txt
./hallucinate.py --chars 200 --input-dir=./input --model ./checkpoints/mybestmodel.hdf5 "once upon a time there was a wild band of marmalade"
```

# TODO

[x] Seed training with an existing model to improve?
[ ] Add more layers into the network with dropout to prevent overfitting - http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
