import pandas as pd
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg
from string import punctuation
from collections import Counter
import numpy as np

def load_data():
    data = gutenberg.raw('shakespeare-hamlet.txt')
    with open('hamlet_data.txt', 'w') as file:
        file.write(data)

    with open('hamlet_data.txt', 'r') as file:
        text = file.read()

    original_text = text
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = ''.join(c.lower() for c in text if c not in punctuation)
    split_text = text.split(' ')
    count_text = Counter(split_text)
    count_text.pop('')
    sorted_text = count_text.most_common()

    tokenized_text = {w : i+1 for i, (w, c) in enumerate(sorted_text)}

    original_text = ''.join(c.lower() for c in original_text if c not in punctuation)
    original_split = original_text.split('\n')
    original_split = [x for x in original_split if x != '']

    input_sequences = []
    for line in original_split:
        token_line = [tokenized_text[word] for word in line.split(' ') if word != '']
        for i in range(1, len(token_line)):
            n_gram_sequence = token_line[:i+1]
            input_sequences.append(n_gram_sequence)
    features = max(len(x) for x in input_sequences)

    padded_sequences = np.zeros((len(input_sequences), features))
    for i in range(len(input_sequences)):
        padded_sequences[i, features-len(input_sequences[i]):] = np.array(input_sequences[i])
    padded_sequences = padded_sequences.astype(int)

    return (padded_sequences, len(tokenized_text)+1, tokenized_text, features)