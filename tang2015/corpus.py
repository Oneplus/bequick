#!/usr/bin/env python
import numpy as np
try:
    import bequick
except ImportError:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from bequick.io_utils import zip_open


def read_and_transform_dataset(filename, alphabet, insert_new_token=False, add_unk_token=False):
    assert alphabet.get('__BAD0__') == 0, "BAD0 should be inserted to dataset before load data."
    documents = []
    raw_document = []

    for line in zip_open(filename):
        line = line.strip()
        if len(line) == 0:
            document = []
            length = []
            label = int(raw_document[0])
            for raw_sentence in raw_document[1:]:
                raw_words = raw_sentence.split()
                sentence = []
                for raw_word in raw_words:
                    if insert_new_token:
                        sentence.append(alphabet.insert(raw_word))
                    else:
                        if raw_word in alphabet:
                            sentence.append(alphabet.get(raw_word))
                        elif add_unk_token:
                            sentence.append(alphabet.get("__UNK__"))
                document.append(sentence)
                length.append(len(sentence))
            documents.append((document, length, label))
            raw_document = []
        else:
            raw_document.append(line)
    return documents


def flatten_dataset(dataset, max_words):
    n_doc = len(dataset)
    X = np.zeros(shape=(n_doc, max_words), dtype=np.int32)
    L = np.zeros(shape=n_doc, dtype=np.int32)
    Y = np.zeros(shape=n_doc, dtype=np.int32)
    for i, (document, length, label) in enumerate(dataset):
        L[i] = sum(length)
        offset = 0
        for sentence in document:
            X[i, offset: offset + len(sentence)] = np.array(sentence, dtype=np.int32)
            offset += len(sentence)
        Y[i] = label
    return X, L, Y


def treelike_dataset(dataset, max_sentences, max_words):
    n_doc = len(dataset)
    X = np.zeros(shape=(n_doc, max_sentences, max_words), dtype=np.int32)
    L = np.ones(shape=(n_doc, max_sentences), dtype=np.int32)  # TRICKY: to avoid divide by zeros.
    L2 = np.zeros(shape=n_doc, dtype=np.int32)
    Y = np.zeros(shape=n_doc, dtype=np.int32)
    for i, (document, length, label) in enumerate(dataset):
        for j, sentence in enumerate(document):
            L[i, j] = len(sentence)
            X[i, j, : len(sentence)] = np.array(sentence, dtype=np.int32)
        Y[i] = label
        L2[i] = len(document)
    return X, L, L2, Y
