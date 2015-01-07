#!/usr/bin/env python
import sys
import gzip
import cPickle as pickle
from scipy.sparse import dok_matrix
from optparse import OptionParser
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

def read_dataset(filename):
    fp = gzip.open(filename, "r")
    dataset = fp.read().strip().split("\n\n")
    X, Y = [], []
    for data in dataset:
        for line in data.split("\n"):
            tokens = line.strip().split()
            label = tokens[0]
            Y.append(label)
            attrs = tokens[1:]
            X.append({attr: 1 for attr in attrs})
    return X, Y

def label_transform(Y, labels, incremental = True):
    for y in Y:
        if y not in labels:
            labels[y] = len(labels)
    return [labels[y] for y in Y]

def learn():
    usage = "the learning component."
    parser = OptionParser(usage)
    parser.add_option("-f", "--feature", dest="feature", help="the feature file.")
    parser.add_option("-m", "--model", dest="model", help="the model file.")
    opts, args = parser.parse_args()
    attrs, labels = DictVectorizer(), {}
    X, Y = read_dataset(opts.feature)
    X = attrs.fit_transform(X)
    Y = label_transform(Y, labels)
    print >> sys.stderr, "trace: #labels", len(labels)
    print >> sys.stderr, "trace: #features", len(attrs.vocabulary_)
    logreg = LogisticRegression(C=1)
    logreg.fit(X, Y)
    print >> sys.stderr, "trace: model learning is done."
    pickle.dump((labels, attrs, logreg), gzip.open(opts.model, "w"))

def test():
    usage = "the testing component."
    parser = OptionParser(usage)
    parser.add_option("-f", "--feature", dest="feature", help="the feature file.")
    parser.add_option("-m", "--model", dest="model", help="the model file.")
    opts, args = parser.parse_args()
    labels, attrs, logreg = pickle.load(gzip.open(opts.model, "r"))
    print >> sys.stderr, "trace: #labels", len(labels)
    print >> sys.stderr, "trace: #features", len(attrs.vocabulary_)
    X, Y = read_dataset(opts.feature)
    X = attrs.transform(X)
    Y = label_transform(Y, labels, False)
    Z = logreg.predict(X)
    nr_correct, nr_total = 0, 0
    for y, z in zip(Y, Z):
        if y == z:
            nr_correct += 1
        nr_total += 1
    print >> sys.stderr, "trace: p=%f" % (float(nr_correct)/nr_total)

if __name__=="__main__":
    usage = "a demo code for using sklearn with maxent chunking."
    if len(sys.argv) <= 1:
        print >> sys.stderr, ("error: %s [learn|test]" % sys.argv[0])
    elif sys.argv[1] == "learn":
        learn()
    elif sys.argv[1] == "test":
        test()
