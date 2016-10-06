#!/usr/bin/env python
# The data preparation script of the SNLI dataset. It
# - extracts sentences from the parse tree.
# - convert the word in raw form into index.
# - save the mapping from word to index into a file.
from __future__ import print_function
import os
import sys
import json
import argparse
import cPickle as pkl
from nltk.tree import Tree

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from bequick.alphabet import Alphabet


def extract_words_from_tree(context):
    tree = Tree.fromstring(context)
    return tree.leaves()


def get_in_files(prefix, suffix):
    return ["%s_%s.%s" % (prefix, split, suffix) for split in ["train", "dev", "test"]]


def convert_and_output(label_alphabet, form_alphabet, gold_label, sentence1, sentence2, fpo, is_train):
    """

    :param label_alphabet: Alphabet
    :param form_alphabet: Alphabet
    :param gold_label: str
    :param sentence1: list(str)
    :param sentence2: list(str)
    :param fpo: file
    :return:
    """
    gold_label_id = label_alphabet.insert(gold_label) if is_train else label_alphabet.get(gold_label)
    sentence1_ids = [(form_alphabet.insert(token) if is_train else form_alphabet.get(token)) for token in sentence1]
    sentence2_ids = [(form_alphabet.insert(token) if is_train else form_alphabet.get(token)) for token in sentence2]
    print('%d\t%s\t%s' % (gold_label_id,
                          ' '.join(str(i) for i in sentence1_ids),
                          ' '.join(str(i) for i in sentence2_ids)), file=fpo)


def parse_json(prefix, label_alphabet, form_alphabet):
    in_files = get_in_files(prefix, "jsonl")
    out_files = get_in_files(prefix, "ext")
    names = ["train", "dev", "test"]

    for in_file, out_file, name in zip(in_files, out_files, names):
        fpo = open(out_file, 'w')
        for line in open(in_file, 'r'):
            payload = json.loads(line)
            gold_label = payload["gold_label"]
            if gold_label == '-': # according to the README.txt, where theres is no majority exists (marked as '-')
                continue
            sentence1 = extract_words_from_tree(payload["sentence1_parse"])
            sentence2 = extract_words_from_tree(payload["sentence2_parse"])
            convert_and_output(label_alphabet, form_alphabet, gold_label, sentence1, sentence2, fpo,
                               True if name is "train" else False)


def parse_tsv(prefix, label_alphabet, form_alphabet):
    in_files = get_in_files(prefix, "txt")
    out_files = get_in_files(prefix, "ext")
    names = ["train", "dev", "test"]

    for in_file, out_file, name in zip(in_files, out_files, names):
        fpo = open(out_file, 'w')
        fpi = open(in_file, 'r')
        fpi.readline()  # skip the header.
        for line in fpi:
            payload = line.strip().split('\t')
            gold_label = payload[0]
            if gold_label == '-':
                continue
            sentence1 = extract_words_from_tree(payload[3])
            sentence2 = extract_words_from_tree(payload[4])
            convert_and_output(label_alphabet, form_alphabet, gold_label, sentence1, sentence2, fpo,
                               True if name is "train" else False)
    pkl.dump(form_alphabet, open("%s_dict.pkl" % prefix, 'wb'))


def main():
    usage = "the data preparation script"
    cmd = argparse.ArgumentParser(usage)
    cmd.add_argument("--json", default=False, action="store_true", help="input is in json.")
    cmd.add_argument("prefix", help="the prefix, file in format of prefix_[train|dev|test].txt")
    args = cmd.parse_args()
    label_alphabet = Alphabet(use_default_initialization=False)
    form_alphabet = Alphabet(use_default_initialization=True)

    if args.json:
        parse_json(args.prefix, label_alphabet, form_alphabet)
    else:
        parse_tsv(args.prefix, label_alphabet, form_alphabet)
    print("loaded %d classes." % len(label_alphabet), file=sys.stderr)
    print("loaded %d forms." % len(form_alphabet), file=sys.stderr)
    pkl.dump(form_alphabet, open("%s_dict.pkl" % args.prefix, 'wb'))


if __name__ == "__main__":
    main()
