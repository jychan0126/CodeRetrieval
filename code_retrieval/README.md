# code_retrieval

## Current situation
Our current model is seq2seq + attension.
(ref https://guillaumegenthial.github.io/sequence-to-sequence.html)

Our current BELU score on conala contest : 27.03	

Top score : 33.51

## Data and code explanation

There are processed corpus data (train and test) in folder **processed_corpus**.

I still want to parse the mined dataset provided by the contest (https://conala-corpus.github.io/), but the current model is trained on the original training set only.

So several things is on the TODO list:

- [x] parse mined data from conala contest. (ichao)
- [ ] use our data in openNMT-pytorch or Tensorflow.
- [ ] try to use state of the art translation method.
- [ ] can we think of new method?
- [ ] do comparison on these methods.
- [ ] add support for partial code as query.
- [ ] combine with user interface.
- [ ] can we provide more than 1 answer?
