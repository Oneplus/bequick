#!/usr/bin/env python
import sys
import argparse
from bowman2015.corpus import Corpus

def main():
    usage = "An implementation of A large annotated corpus for learning natural language inference"
    cmd = argparse.ArgumentParser(usage=usage)
    cmd.add_argument("train", help="the path to the training file.")
    cmd.add_argument("devel", help="the path to the development file.")
    cmd.add_argument("test", help="the path to the testing file.")
    args = cmd.parse_args()

    corpus = Corpus()
    corpus.load_training_data(args.train)
    corpus.load_test_data(args.devel)
    corpus.load_test_data(args.devel)

if __name__ == "__main__":
    main()