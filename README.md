http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

# Training

`--input-dir` is used to read input text files in that directory to train on. No validation datasets are used; we train on everything to have the most data for character occurrence. See `--help` for details on parameters.

```
source bin/activate
pip install -r requirements.txt
./train.py
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

[ ] Seed training with an existing model to improve?
