#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import argparse
import pickle as pkl
try:
    from bequick.utils import zip_open
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
    from bequick.utils import zip_open


def main():
    usage = "Converting raw form word embedding into indexed word embedding."
    cmd = argparse.ArgumentParser(usage=usage)
    cmd.add_argument("dictionary", help="the path to the dictionary.")
    cmd.add_argument("raw_embeddings", help="the path to the raw embedding.")
    args = cmd.parse_args()
    form_alphabet = pkl.load(open(args.dictionary, 'rb'))
    print("alphabet is loaded", file=sys.stderr)

    header_loaded = False
    for line in zip_open(args.raw_embeddings):
        tokens = line.strip().split()
        if not header_loaded and len(tokens) == 2: # preserve the header.
            header_loaded = True
            print(line.strip())
            continue
        word = tokens[0]
        if word in form_alphabet:
            idx = form_alphabet.get(word)
            tokens[0] = str(idx)
            print(' '.join(tokens))

if __name__ == "__main__":
    main()
