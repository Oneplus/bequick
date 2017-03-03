#!/usr/bin/env python
from __future__ import print_function
import argparse
import logging
import itertools
import numpy as np
try:
    import bequick
except ImportError:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from bequick.alphabet import Alphabet
from bequick.embedding import load_embedding_and_build_alphabet
from tang2015.corpus import read_and_transform_dataset
from tang2015.model import FlattenAverage, FlattenBiLSTM
np.random.seed(1234)
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
LOG = logging.getLogger('tang2015')


def evaluate(dataset, model):
    n_corr, n_total = 0, 0
    for X, Y in dataset:
        pred = model.classify(X)
        if pred == Y:
            n_corr += 1
        n_total += 1
    return float(n_corr) / n_total


def main():
    cmd = argparse.ArgumentParser("An implementation of Tang et al. (2015)")
    cmd.add_argument("--model", required=True, help="The model [flat_avg, flat_bilstm]")
    cmd.add_argument("--form_dim", type=int, required=True, help="the dim of the form.")
    cmd.add_argument("--hidden_dim", type=int, required=True, help="the dim of the hidden output.")
    cmd.add_argument("--layers", type=int, default=1, help='the number of layers.')
    cmd.add_argument("--batch_size", type=int, default=32, help='the batch size.')
    cmd.add_argument("--algorithm", default="adam",
                     help="the algorithm [clipping_sgd, adagrad, adadelta, adam].")
    cmd.add_argument("--embedding", help="the path to the word embedding.")
    cmd.add_argument("--epoch", type=int, default=10, help="the epoch.")
    cmd.add_argument("train", default="snli_1.0_train.jsonl", help="the path to the training file.")
    cmd.add_argument("devel", default="snli_1.0_dev.jsonl", help="the path to the development file.")
    cmd.add_argument("test", default="snli_1.0_test.jsonl", help="the path to the testing file.")
    args = cmd.parse_args()

    alphabet = Alphabet(use_default_initialization=True)
    indices, embeddings = load_embedding_and_build_alphabet(args.embedding, alphabet, args.form_dim)
    LOG.info("loaded {0} embedding entries".format(embeddings.shape[0]))
    train_set = read_and_transform_dataset(args.train, alphabet, False, False)
    devel_set = read_and_transform_dataset(args.devel, alphabet, False, False)
    test_set = read_and_transform_dataset(args.test, alphabet, False, False)

    form_size = len(alphabet)
    LOG.info("vocabulary size: {0}".format(form_size))
    n_classes, max_steps = 0, 0
    for X, Y in itertools.chain(train_set, devel_set, test_set):
        flatten_X = [item for sublist in X for item in sublist]
        max_steps = max(max_steps, len(flatten_X))
        n_classes = max(n_classes, Y + 1)
    LOG.info("number of classes: {0}".format(n_classes))
    LOG.info("max steps: {0}".format(max_steps))

    if args.model == 'flat_avg':
        model = FlattenAverage(algorithm=args.algorithm,
                               form_size=form_size,
                               form_dim=args.form_dim,
                               hidden_dim=args.hidden_dim,
                               output_dim=n_classes,
                               max_steps=max_steps)
    else:
        model = FlattenBiLSTM(algorithm=args.algorithm,
                              form_size=form_size,
                              form_dim=args.form_dim,
                              hidden_dim=args.hidden_dim,
                              output_dim=n_classes,
                              max_steps=max_steps)
    model.initialize_word_embeddings(indices, embeddings)
    LOG.info("word embedding initialized.")

    n_train, n_devel, n_test = len(train_set), len(devel_set), len(test_set)
    order = np.arange(n_train, dtype=np.int32)

    best_dev_p, test_p = None, None
    for e in range(args.epoch):
        np.random.shuffle(order)
        cost = 0.
        for o in order:
            X, Y = train_set[o]
            cost += model.train(X, Y)
        LOG.info("cost after iteration {0}: {1:.4f}".format(e, cost))
        p = evaluate(devel_set, model)
        if best_dev_p is None or best_dev_p < p:
            best_dev_p = p
            test_p = evaluate(test_set, model)
            LOG.info("new best on devel is achieved: {0:.4f}, test: {1:.4f}".format(best_dev_p, test_p))
    LOG.info("training done, best devel p: {0:.4f}, test p: {1:.4f}".format(best_dev_p, test_p))


if __name__ == "__main__":
    main()
