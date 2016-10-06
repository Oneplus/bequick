#!/usr/bin/env python
import os
import sys
import argparse
import logging
import random
import numpy as np
from itertools import chain
from corpus import read_dataset, get_alphabet
from model import Model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from bequick.utils import batch

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


def load_embedding(path, form_alphabet, dim):
    indices = []
    matrix = np.zeros(shape=(len(form_alphabet), dim))
    row = 0
    for line in open(path, 'r'):
        tokens = line.strip().split()
        word = tokens[0]
        if word in form_alphabet:
            key = form_alphabet.get(word)
            indices.append(key)
            matrix[row,:]= np.array([float(x) for x in tokens[1:]])
            row += 1
    return indices, matrix[:row, :]


def transduce(chunk, form_alphabet, pos_alphabet, batch_size, max_steps):
    """
    Transduce a batch of data into numeric form.
    :param chunk: list, a list of data
    :param form_alphabet: dict, the form alphabet.
    :param pos_alphabet: dict, the postag alphabet.
    :param batch_size:
    :param max_steps:
    :return:
    """
    stops = [0 for _ in range(batch_size)]
    X = np.zeros(shape=(batch_size, max_steps), dtype=np.int32)
    Y = np.zeros(shape=(batch_size, max_steps), dtype=np.int32)
    for b, data in enumerate(chunk):
        for i, token in enumerate(data):
            X[b, i] = form_alphabet.get(token['form'], 1)
            Y[b, i] = pos_alphabet.get(token['pos'])
        stops[b] = len(data)
    return X, Y, stops


def evaluate(dataset, form_alphabet, pos_alphabet, model, batch_size, max_steps):
    """

    :param dataset:
    :param form_alphabet:
    :param pos_alphabet:
    :param model:
    :param batch_size:
    :param max_steps:
    :return: float, the precision
    """
    n_corr, n_total = 0, 0
    for chunk in batch(dataset):
        X, Y, stops = transduce(chunk, form_alphabet, pos_alphabet, batch_size, max_steps)
        Y_pred = model.classify(X, stops)
        for b in range(len(chunk)):
            for i in range(stops[b]):
                if np.argmax(Y_pred[b, i]) == Y[b, i]:
                    n_corr += 1
                n_total += 1
    return float(n_corr) / n_total


def get_max_steps(train_data, devel_data, test_data):
    max_steps = 0
    for data in chain(train_data, devel_data, test_data):
        max_steps = max(max_steps, len(data))
    return max_steps


def main():
    usage = "A implementation of the Bidirectional LSTM labeler."
    cmd = argparse.ArgumentParser(usage=usage)
    cmd.add_argument("--model", help="The path to the model.")
    cmd.add_argument("--embedding", help="The path to the embedding file.")
    cmd.add_argument("--max-iter", dest="max_iter",type=int, default=10, help="The number of max iteration.")
    cmd.add_argument("--layers", dest="layers", type=int, default=2, help="The size of layers.")
    cmd.add_argument("--hidden-size", dest="hidden_size", type=int, default=200, help="The size of hidden layer.")
    cmd.add_argument("--embedding-size", dest="embedding_size", type=int, default=100, help="The size of embedding.")
    cmd.add_argument("--evaluate-stops", dest="evaluate_stops", type=int, default=-1, help="evaluate each n sentence.")
    cmd.add_argument("--report-stops", dest="report_stops", type=int, default=-1, help="evaluate each n sentence.")
    cmd.add_argument("--lambda", dest="lambda_", type=float, default=1e-8, help="The regularizer parameter.")
    cmd.add_argument("--dropout", dest="dropout", type=float, default=0.5, help="The probability for dropout.")
    cmd.add_argument("--algorithm", default="clipping_sgd", help="The learning algorithm.")
    cmd.add_argument("--batch-size", dest="batch_size", type=int, default=32, help="The batch size.")
    cmd.add_argument("reference", help="The path to the reference file.")
    cmd.add_argument("development", help="The path to the development file.")
    cmd.add_argument("test", help="The path to the test file.")
    opts = cmd.parse_args()

    train_dataset = read_dataset(opts.reference)
    devel_dataset = read_dataset(opts.development)
    test_dataset = read_dataset(opts.test)
    form_alphabet = get_alphabet(train_dataset, 'form')
    pos_alphabet = get_alphabet(train_dataset, 'pos')
    logging.info("# training data: %d" % len(train_dataset))
    logging.info("# development data: %d" % len(devel_dataset))
    logging.info("# test data: %d" % len(test_dataset))
    logging.info("# form alpha: %d" % len(form_alphabet))
    logging.info("# pos alpha: %d" % len(pos_alphabet))

    max_steps = get_max_steps(train_dataset, devel_dataset, test_dataset)
    logging.info('max steps %d' % max_steps)

    model = Model(algorithm=opts.algorithm,
                  form_size=len(form_alphabet),
                  form_dim=opts.embedding_size,
                  hidden_dim=opts.hidden_size,
                  output_dim=len(pos_alphabet),   # should also counting the BAD0, because
                  n_layers=opts.layers,
                  max_steps=max_steps,
                  batch_size=opts.batch_size,
                  regularizer=opts.lambda_)

    model.init()
    if opts.embedding is not None:
        indices, matrix = load_embedding(opts.embedding, form_alphabet, opts.embedding_size)
        model.initialize_word_embeddings(indices, matrix)
        logging.info('Embedding is loaded.')


    n_sentence = 0
    best_p, test_score = 0., 0.

    for iteration in range(1, opts.max_iter + 1):
        logging.info(('Iteration %d' % iteration))
        random.shuffle(train_dataset)
        cost = 0.
        for chunk in batch(train_dataset, opts.batch_size):
            X, Y, stops = transduce(chunk, form_alphabet, pos_alphabet, opts.batch_size, max_steps)
            cost += model.train(X, Y, stops)

            n_sentence += 1
            if opts.report_stops > 0 and (n_sentence % opts.report_stops == 0):
                logging.info('Processed %d sentence.' % n_sentence)
            if opts.evaluate_stops > 0 and n_sentence % opts.evaluate_stops == 0:
                logging.info('Finish %f sentences' % (float(n_sentence) / len(train_dataset)))
                p = evaluate(devel_dataset, form_alphabet, pos_alphabet, model, opts.batch_size, max_steps)
                logging.info('Development score is: %f' % p)
                if p > best_p:
                    best_p = p
                    logging.info('New best score achieved!')
                    test_score = evaluate(test_dataset, form_alphabet, pos_alphabet, model, opts.batch_size, max_steps)
                    logging.info('Test score is: %f' % test_score)

        logging.info('Iteration %d: Cost %f' % (iteration, cost))
        p = evaluate(devel_dataset, form_alphabet, pos_alphabet, model, opts.batch_size, max_steps)
        logging.info('Devel at the end of iteration %d, accuracy=%f' % (iteration, p))
        if p > best_p:
            best_p = p
            logging.info('New best achieved: %f' % best_p)
            test_score = evaluate(test_dataset, form_alphabet, pos_alphabet, model, opts.batch_size, max_steps)
            logging.info('Test score is: %f' % test_score)

    logging.info('Finish training, best development accuracy is %f, test accuracy is %f' % (best_p, test_score))


if __name__ == "__main__":
    main()
