# Next Word Prediction using GRU

A deep learning project for predicting the next word in a sequence
using a recurrent neural network trained on Shakespeare's *Hamlet*.

## Overview

Next-word prediction is a fundamental Natural Language Processing
(NLP) task used in applications such as predictive text, autocomplete,
and language modeling.

This project develops a neural language model that learns patterns and
word sequences from the text of Shakespeare's *Hamlet* and predicts
the next word given a sequence of preceding words.

The model is implemented using PyTorch and uses an embedding layer
followed by a Gated Recurrent Unit (GRU) network.

## Dataset

The model is trained on Shakespeare's *Hamlet*, obtained through the
NLTK Gutenberg corpus.

The preprocessing pipeline:

1. Loads the *Hamlet* text using NLTK.
2. Removes punctuation and normalizes text to lowercase.
3. Tokenizes the text into individual words.
4. Builds a vocabulary based on word frequency.
5. Maps each word to a unique integer index.
6. Generates sequential n-gram training examples.
7. Pads the sequences to a common length for model training.

The processed corpus is stored in `hamlet_data.txt`.

## Model Architecture

The model uses the following architecture:

```text
Input Word Sequence
        │
        ▼
Embedding Layer
        │
        ▼
GRU Layer
        │
        ▼
Dropout
        │
        ▼
Fully Connected Layer
        │
        ▼
Next-Word Prediction
