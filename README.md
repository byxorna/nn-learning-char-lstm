http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

# Running

`INPUT_DIR=` is used to read input text files in that directory to train on. No validation datasets are used; we train on everything to have the most data for character occurrence.
```
source bin/activate
pip install -r requirements.txt
./train.py
```

# Hallucinating

```
source bin/activate
pip install -r requirements.txt
./hallucinate.py --chars 200 "once upon a time there was a wild band of marmalade"
```
