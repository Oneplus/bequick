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
    documents = []
    raw_document = []

    for line in zip_open(filename):
        line = line.strip()
        if len(line) == 0:
            document = []
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
            documents.append((document, label))
            raw_document = []
        else:
            raw_document.append(line)
    return documents
