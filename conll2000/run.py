#!/usr/bin/env python
from __future__ import print_function
import gzip
import pickle
from optparse import OptionParser
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
try:
    import bequick
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from bequick.io_utils import zip_open
try:
    from .conlleval import evaluate, report
except (ValueError, SystemError) as e:
    from conlleval import evaluate, report


def read_dataset(filename):
    fp = zip_open(filename)
    dataset = fp.read().decode().strip().split("\n\n")
    X, Y = [], []
    for data in dataset:
        for line in data.split("\n"):
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            label = tokens[0]
            Y.append(label)
            X.append({attr: 1 for attr in tokens[1:]})
    return X, Y


def label_transform(Y, labels, incremental=True):
    for y in Y:
        if y not in labels:
            labels[y] = len(labels)
    return [labels[y] for y in Y]


def learn():
    cmd = OptionParser("the learning component.")
    cmd.add_option("-f", "--feature", dest="feature", help="the feature file.")
    cmd.add_option("-m", "--model", dest="model", help="the model file.")
    opts, args = cmd.parse_args()
    attrs, labels = DictVectorizer(), {}
    X, Y = read_dataset(opts.feature)
    X = attrs.fit_transform(X)
    Y = label_transform(Y, labels)
    print("trace: #labels {0}".format(len(labels)), file=sys.stderr)
    print("trace: #features {0}".format(len(attrs.vocabulary_)), file=sys.stderr)
    logreg = LogisticRegression(C=1)
    logreg.fit(X, Y)
    print("trace: model learning is done.", file=sys.stderr)
    pickle.dump((labels, attrs, logreg), gzip.open(opts.model, "w"))


def test():
    cmd = OptionParser("the testing component.")
    cmd.add_option("-f", "--feature", dest="feature", help="the feature file.")
    cmd.add_option("-m", "--model", dest="model", help="the model file.")
    opts, args = cmd.parse_args()
    labels, attrs, logreg = pickle.load(gzip.open(opts.model, "r"))
    print("trace: #labels {0}".format(len(labels)), file=sys.stderr)
    print("trace: #features {0}".format(len(attrs.vocabulary_)), file=sys.stderr)
    X, Y = read_dataset(opts.feature)
    X = attrs.transform(X)
    Y = label_transform(Y, labels, False)
    Z = logreg.predict(X)
    print("trace: p={0}".format(accuracy_score(Y, Z)), file=sys.stderr)

    inverse_labels = {}
    for key, value in labels.items():
        inverse_labels[value] = key
    bundle = ["_ _ {0} {1}".format(inverse_labels[z], inverse_labels[y]) for z, y in zip(Z, Y)]
    report(evaluate(bundle))

if __name__ == "__main__":
    usage = "a demo code for using sklearn with maxent chunking."
    if len(sys.argv) <= 1:
        print("error: %s [learn|test]" % sys.argv[0], file=sys.stderr)
    elif sys.argv[1] == "learn":
        learn()
    elif sys.argv[1] == "test":
        test()
