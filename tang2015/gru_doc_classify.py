#!/usr/bin/env python
from __future__ import print_function
import argparse
import logging
import itertools
import numpy as np
import sklearn.metrics
try:
    import bequick
except ImportError:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from bequick.alphabet import Alphabet
from bequick.embedding import load_embedding_and_build_alphabet
try:
    from .corpus import read_and_transform_dataset, flatten_dataset, treelike_dataset
    from .model import (FlattenBiLSTM, FlattenAverage, FlattenBiGRU, TreeAveragePipeGRU, TreeGRUPipeAverage,
                        TreeGRUPipeGRU)
except (ValueError, SystemError) as e:
    from corpus import read_and_transform_dataset, flatten_dataset, treelike_dataset
    from model import (FlattenBiLSTM, FlattenAverage, FlattenBiGRU, TreeAveragePipeGRU, TreeGRUPipeAverage,
                        TreeGRUPipeGRU)
np.random.seed(1234)
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
LOG = logging.getLogger('tang2015')


def evaluate(package, model, n_classes, batch_size):
    if len(package) == 3:
        X, L, Y = package
    elif len(package) == 4:
        X, L, L2, Y = package
    else:
        assert False
    n = Y.shape[0]
    prediction = np.full(n, n_classes + 1, dtype=np.int32)
    for batch_start in range(0, n, batch_size):
        batch_end = batch_start + batch_size if batch_start + batch_size < n else n
        if len(package) == 3:
            prediction[batch_start: batch_end] = model.classify(X[batch_start: batch_end],
                                                                L[batch_start: batch_end])
        else:
            prediction[batch_start: batch_end] = model.classify(X[batch_start: batch_end],
                                                                L[batch_start: batch_end],
                                                                L2[batch_start: batch_end])
    return sklearn.metrics.accuracy_score(Y, prediction)


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
    cmd.add_argument("--epochs", type=int, default=10, help="the epoch.")
    cmd.add_argument("--report_stops", type=int, default=-1, help="The number of stops")
    cmd.add_argument("train", help="the path to the training file.")
    cmd.add_argument("devel", help="the path to the development file.")
    cmd.add_argument("test", help="the path to the testing file.")
    args = cmd.parse_args()

    alphabet = Alphabet(use_default_initialization=True)
    indices, embeddings = load_embedding_and_build_alphabet(args.embedding, alphabet, args.form_dim)
    LOG.info("loaded {0} embedding entries".format(embeddings.shape[0]))
    train_set = read_and_transform_dataset(args.train, alphabet, False, False)
    devel_set = read_and_transform_dataset(args.devel, alphabet, False, False)
    test_set = read_and_transform_dataset(args.test, alphabet, False, False)
    LOG.info("dataset is loaded: # train={0}, # dev={1}, # test={2}".format(len(train_set), len(devel_set),
                                                                            len(test_set)))
    form_size = len(alphabet)
    LOG.info("vocabulary size: {0}".format(form_size))
    use_flatten_model = args.model.startswith('flat')

    if use_flatten_model:
        n_classes, max_steps = 0, 0
        for X, L, Y in itertools.chain(train_set, devel_set, test_set):
            max_steps = max(max_steps, sum(L))
            n_classes = max(n_classes, Y + 1)
        LOG.info("number of classes: {0}".format(n_classes))
        LOG.info("max steps: {0}".format(max_steps))

        train_X, train_L, train_Y = flatten_dataset(train_set, max_steps)
        devel_X, devel_L, devel_Y = flatten_dataset(devel_set, max_steps)
        test_X, test_L, test_Y = flatten_dataset(test_set, max_steps)
        LOG.info("dataset is flattened.")

        if args.model == 'flat_avg':
            model = FlattenAverage(algorithm=args.algorithm,
                                   form_size=form_size, form_dim=args.form_dim, hidden_dim=args.hidden_dim,
                                   output_dim=n_classes,
                                   max_steps=max_steps,
                                   batch_size=args.batch_size)
        elif args.model == 'flat_bigru':
            model = FlattenBiGRU(algorithm=args.algorithm, n_layers=1,
                                 form_size=form_size, form_dim=args.form_dim, hidden_dim=args.hidden_dim,
                                 output_dim=n_classes,
                                 max_steps=max_steps,
                                 batch_size=args.batch_size)
        else:
            model = FlattenBiLSTM(algorithm=args.algorithm, n_layers=1,
                                  form_size=form_size, form_dim=args.form_dim, hidden_dim=args.hidden_dim,
                                  output_dim=n_classes,
                                  max_steps=max_steps,
                                  batch_size=args.batch_size)
    else:
        n_classes, max_sentences, max_words = 0, 0, 0
        for X, L, Y in itertools.chain(train_set, devel_set, test_set):
            max_sentences = max(max_sentences, len(X))
            max_words = max(max_words, np.max(L))
            n_classes = max(n_classes, Y + 1)
        LOG.info("number of classes: {0}".format(n_classes))
        LOG.info("max number of sentences: {0}".format(max_sentences))
        LOG.info("max number of words: {0}".format(max_words))

        train_X, train_L, train_L2, train_Y = treelike_dataset(train_set, max_sentences, max_words)
        devel_X, devel_L, devel_L2, devel_Y = treelike_dataset(devel_set, max_sentences, max_words)
        test_X, test_L, test_L2, test_Y = treelike_dataset(test_set, max_sentences, max_words)
        LOG.info("dataset is flattened.")

        if args.model == 'tree_avg_gru':
            model = TreeAveragePipeGRU(algorithm=args.algorithm, n_layers=1,
                                       form_size=form_size, form_dim=args.form_dim, hidden_dim=args.hidden_dim,
                                       output_dim=n_classes,
                                       max_sentences=max_sentences, max_words=max_words,
                                       batch_size=args.batch_size)
        elif args.model == 'tree_gru_avg':
            model = TreeGRUPipeAverage(algorithm=args.algorithm, n_layers=1,
                                       form_size=form_size, form_dim=args.form_dim, hidden_dim=args.hidden_dim,
                                       output_dim=n_classes,
                                       max_sentences=max_sentences, max_words=max_words,
                                       batch_size=args.batch_size)
        else:
            model = TreeGRUPipeGRU(algorithm=args.algorithm, n_layers=1,
                                   form_size=form_size, form_dim=args.form_dim, hidden_dim=args.hidden_dim,
                                   output_dim=n_classes,
                                   max_sentences=max_sentences, max_words=max_words,
                                   batch_size=args.batch_size)
    model.initialize_word_embeddings(indices, embeddings)
    LOG.info("word embedding initialized.")

    n_train, n_devel, n_test = train_Y.shape[0], devel_Y.shape[0], test_Y.shape[0]
    order = np.arange(n_train, dtype=np.int32)

    best_dev_p, test_p = None, None
    n_stops = 0
    for epoch in range(args.epochs):
        LOG.info("start iteration #{0}".format(epoch))
        np.random.shuffle(order)
        cost = 0.
        for batch_start in range(0, n_train, args.batch_size):
            batch_end = batch_start + args.batch_size if batch_start + args.batch_size < n_train else n_train
            batch = order[batch_start: batch_end]
            if use_flatten_model:
                X, L, Y = train_X[batch], train_L[batch], train_Y[batch]
                cost += model.train(X, L, Y)
            else:
                X, L, L2, Y = train_X[batch], train_L[batch], train_L2[batch], train_Y[batch]
                cost += model.train(X, L, L2, Y)
            n_stops += 1
            if args.report_stops > 0 and n_stops % args.report_stops == 0:
                package = (devel_X, devel_L, devel_Y) if use_flatten_model else (
                    devel_X, devel_L, devel_L2, devel_Y)
                p = evaluate(package, model, n_classes, args.batch_size)
                LOG.info("evaluate on #{0}-th batches".format(n_stops / args.report_stops))
                if best_dev_p is None or best_dev_p < p:
                    best_dev_p = p
                    package = (test_X, test_L, test_Y) if use_flatten_model else (
                        test_X, test_L, test_L2, test_Y)
                    test_p = evaluate(package, model, n_classes, args.batch_size)
                    LOG.info("new best on devel is achieved: {0:.4f}, test: {1:.4f}".format(best_dev_p, test_p))
        package = (devel_X, devel_L, devel_Y) if use_flatten_model else (devel_X, devel_L, devel_L2, devel_Y)
        p = evaluate(package, model, n_classes, args.batch_size)
        LOG.info("after iteration #{0}: cost={1:.4f}, devp={2:.4f}".format(epoch, cost, p))
        if best_dev_p is None or best_dev_p < p:
            best_dev_p = p
            package = (test_X, test_L, test_Y) if use_flatten_model else (test_X, test_L, test_L2, test_Y)
            test_p = evaluate(package, model, n_classes, args.batch_size)
            LOG.info("new best on devel is achieved: {0:.4f}, test: {1:.4f}".format(best_dev_p, test_p))
    LOG.info("training done, best devel p: {0:.4f}, test p: {1:.4f}".format(best_dev_p, test_p))


if __name__ == "__main__":
    main()
