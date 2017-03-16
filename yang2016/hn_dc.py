#!/usr/bin/env python
from __future__ import print_function
import argparse
import logging
import itertools
import numpy as np
import tensorflow as tf
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
    from .corpus import read_and_transform_dataset, treelike_dataset
    from .model import HN_AVE, HN_MAX, HN_ATT
except (ValueError, SystemError) as e:
    from corpus import read_and_transform_dataset, treelike_dataset
    from model import HN_AVE, HN_MAX, HN_ATT
np.random.seed(1234)
tf.set_random_seed(1234)
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
LOG = logging.getLogger('tang2015')


def evaluate(session, package, model, n_classes, batch_size):
    X, L, L2, Y = package
    n = Y.shape[0]
    prediction = np.full(n, n_classes + 1, dtype=np.int32)
    for batch_start in range(0, n, batch_size):
        batch_end = batch_start + batch_size if batch_start + batch_size < n else n
        prediction[batch_start: batch_end] = model.classify(session,
                                                            X[batch_start: batch_end],
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
    cmd.add_argument("--max_sentences", type=int, default=100, help="The maximum number of sentences.")
    cmd.add_argument("--max_words", type=int, default=150, help="The maximum number of words in sentence.")
    cmd.add_argument("--debug", default=False, action="store_true", help="Use to specify debug.")
    cmd.add_argument('--tune_embedding', default=False, action="store_true",
                     help="Use to specify whether tune embedding.")
    cmd.add_argument("train", help="the path to the training file.")
    cmd.add_argument("devel", help="the path to the development file.")
    cmd.add_argument("test", help="the path to the testing file.")
    args = cmd.parse_args()

    alphabet = Alphabet(use_default_initialization=True)
    indices, embeddings = load_embedding_and_build_alphabet(args.embedding, alphabet, args.form_dim)
    LOG.info("loaded {0} embedding entries".format(embeddings.shape[0]))
    train_set = read_and_transform_dataset(args.train, alphabet, args.max_sentences, args.max_words,
                                           args.tune_embedding, False)
    devel_set = read_and_transform_dataset(args.devel, alphabet, args.max_sentences, args.max_words, False, False)
    test_set = read_and_transform_dataset(args.test, alphabet, args.max_sentences, args.max_words, False, False)
    LOG.info("dataset is loaded: # train={0}, # dev={1}, # test={2}".format(len(train_set), len(devel_set),
                                                                            len(test_set)))
    form_size = len(alphabet)
    LOG.info("vocabulary size: {0}".format(form_size))

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
    LOG.info("dataset is transformed.")

    kwargs = {'algorithm': args.algorithm, 'form_size': form_size, 'form_dim': args.form_dim,
              'hidden_dim': args.hidden_dim, 'output_dim': n_classes, 'max_sentences': max_sentences,
              'max_words': max_words, 'batch_size': args.batch_size, 'tune_embedding': args.tune_embedding,
              'debug': args.debug, 'n_layers': 1}
    if args.model == 'hn_avg':
        model = HN_AVE(**kwargs)
    elif args.model == 'hn_max':
        model = HN_MAX(**kwargs)
    else:
        model = HN_ATT(**kwargs)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    # session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    if args.debug:
        summary_writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph())

    model.initialize_word_embeddings(session, indices, embeddings)
    LOG.info("word embedding initialized.")

    n_train, n_devel, n_test = train_Y.shape[0], devel_Y.shape[0], test_Y.shape[0]
    order = np.arange(n_train, dtype=np.int32)

    best_dev_p, test_p = None, None
    n_stops = 0
    for epoch in range(args.epochs):
        LOG.info("start iteration #{0}".format(epoch))
        np.random.shuffle(order)
        cost, acc = 0., 0.
        for batch_start in range(0, n_train, args.batch_size):
            batch_end = batch_start + args.batch_size if batch_start + args.batch_size < n_train else n_train
            batch = order[batch_start: batch_end]
            kwargs = {'session': session, 'documents': train_X[batch], 'labels': train_Y[batch],
                      'lengths': train_L[batch], 'lengths2': train_L2[batch]}
            if n_stops % 100 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                kwargs.update({'run_options': run_options, 'run_metadata': run_metadata})
            ret = model.train(**kwargs)
            cost += ret[0]
            acc += ret[1] * batch.shape[0]
            if args.debug:
                if n_stops % 100 == 0:
                    summary_writer.add_run_metadata(run_metadata, 'step{0}'.format(n_stops))
                summary_writer.add_summary(ret[2], n_stops)

            n_stops += 1
            if args.report_stops > 0 and n_stops % args.report_stops == 0:
                package = devel_X, devel_L, devel_L2, devel_Y
                p = evaluate(session, package, model, n_classes, args.batch_size)
                LOG.info("evaluate on #{0}-th batches".format(n_stops / args.report_stops))
                if best_dev_p is None or best_dev_p < p:
                    best_dev_p = p
                    package = test_X, test_L, test_L2, test_Y
                    test_p = evaluate(session, package, model, n_classes, args.batch_size)
                    LOG.info("new best on devel is achieved: {0:.4f}, test: {1:.4f}".format(best_dev_p, test_p))
        package = devel_X, devel_L, devel_L2, devel_Y
        p = evaluate(session, package, model, n_classes, args.batch_size)
        LOG.info("after iteration #{0}: cost={1:.4f}, acc={2:.4f}, devp={3:.4f}".format(epoch, cost,
                                                                                        float(acc / n_train), p))
        if best_dev_p is None or best_dev_p < p:
            best_dev_p = p
            package = test_X, test_L, test_L2, test_Y
            test_p = evaluate(session, package, model, n_classes, args.batch_size)
            LOG.info("new best on devel is achieved: {0:.4f}, test: {1:.4f}".format(best_dev_p, test_p))
    LOG.info("training done, best devel p: {0:.4f}, test p: {1:.4f}".format(best_dev_p, test_p))


if __name__ == "__main__":
    main()
