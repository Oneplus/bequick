import numpy as np


def load_embedding(path, form_alphabet, dim, set_none_to_zeros=False):
    indices = []
    matrix = np.zeros(shape=(len(form_alphabet), dim), dtype=np.float32)
    row = 0
    for line in open(path, 'r', encoding='utf-8'):
        tokens = line.strip().split()
        word = tokens[0]
        if word in form_alphabet:
            key = form_alphabet.get(word)
            indices.append(key)
            matrix[row, :] = np.array([float(x) for x in tokens[1:]], dtype=np.float32)
            row += 1
    if set_none_to_zeros:
        indices.append(0)
        matrix[row, :] = np.zeros(dim, dtype=np.float32)
        row += 1
    return indices, matrix[:row, :]
