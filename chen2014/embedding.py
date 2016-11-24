import numpy as np


def load_embedding(path, form_alphabet, dim):
    indices = []
    matrix = np.zeros(shape=(len(form_alphabet), dim))
    row = 0
    for line in open(path, 'r'):
        tokens = line.strip().split()
        word = tokens[0]
        if word in form_alphabet:
            key = form_alphabet.get(word)
            indices.append(key)
            matrix[row, :] = np.array([float(x) for x in tokens[1:]])
            row += 1
    return indices, matrix[:row, :]
