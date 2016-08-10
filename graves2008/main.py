#!/usr/bin/env python
import sys
import argparse
import logging
import random
import numpy as np
from corpus import read_dataset, get_alphabet
from model import ModelTF9, ModelTF10rc

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


def evaluate(dataset, form_alphabet, pos_alphabet, model, max_steps):
    n_corr, n_total = 0, 0
    for data in dataset:
        X = np.zeros(max_steps, dtype=np.int32)
        for i, token in enumerate(data):
            X[i] = form_alphabet.get(token['form'], 1)
        Y = model.classify(X, len(data))
        for y, token in zip(Y, data):
            y_gold = token['pos']
            if np.argmax(y) == pos_alphabet.get(y_gold) - 2:
                n_corr += 1
            n_total += 1
    return float(n_corr) / n_total


def learn():
    cmd = argparse.ArgumentParser("Learning component for chen and manning (2014)'s parser")
    cmd.add_argument("--model", help="The path to the model.")
    cmd.add_argument("--embedding", help="The path to the embedding file.")
    cmd.add_argument("--reference", help="The path to the reference file.")
    cmd.add_argument("--development", help="The path to the development file.")
    cmd.add_argument("--max-iter", dest="max_iter",type=int, default=10,
                      help="The number of max iteration.")
    cmd.add_argument("--layers", dest="layers", type=int, default=2, help="The size of layers.")
    cmd.add_argument("--hidden-size", dest="hidden_size", type=int, default=200, help="The size of hidden layer.")
    cmd.add_argument("--embedding-size", dest="embedding_size", type=int, default=100, help="The size of embedding.")
    cmd.add_argument("--evaluate-stops", dest="evaluate_stops", type=int, default=-1, help="evaluate each n sentence.")
    cmd.add_argument("--report-stops", dest="report_stops", type=int, default=-1, help="evaluate each n sentence.")
    cmd.add_argument("--lambda", dest="lambda", type=float, default=1e-8, help="The regularizer parameter.")
    cmd.add_argument("--dropout", dest="dropout", type=float, default=0.5, help="The probability for dropout.")
    cmd.add_argument("--tf-version", dest="tf_ver", type=str, default="10rc", help="The version.")
    opts = cmd.parse_args(sys.argv[2:])

    train_dataset = read_dataset(opts.reference)
    devel_dataset = read_dataset(opts.development)
    form_alphabet = get_alphabet(train_dataset, 'form')
    pos_alphabet = get_alphabet(train_dataset, 'pos')
    logging.info("# training data: %d" % len(train_dataset))
    logging.info("# development data: %d" % len(devel_dataset))
    logging.info("# form alpha: %d" % len(form_alphabet))
    logging.info("# pos alpha: %d" % len(pos_alphabet))

    max_steps = max(
            max([len(data) for data in train_dataset]),
            max([len(data) for data in devel_dataset])
            )
    logging.info('max steps %d' % max_steps)

    if opts.tf_ver == "10rc":
        # NOT FINISHED
        model = ModelTF10rc(
                form_size=len(form_alphabet),
                form_dim=opts.embedding_size,
                hidden_dim=opts.hidden_size,
                output_dim=(len(pos_alphabet) - 2),
                n_layers=opts.layers,
                max_steps=max_steps
                )
    else:
        model = ModelTF9(
                form_size=len(form_alphabet),
                form_dim=opts.embedding_size,
                hidden_dim=opts.hidden_size,
                output_dim=(len(pos_alphabet) - 2),
                n_layers=opts.layers,
                max_steps=max_steps
                )
    model.init()
    indices, matrix = load_embedding(opts.embedding, form_alphabet, opts.embedding_size)
    model.initialize_word_embeddings(indices, matrix)
    logging.info('Embedding is loaded.')

    n_sentence = 0
    best_p = 0
    for iter in range(1, opts.max_iter + 1):
        logging.info(('Iteration %d' % iter))
        order = range(len(train_dataset))
        random.shuffle(order)
        cost = 0.
        for n in order:
            train_data = train_dataset[n]
            X = np.zeros(max_steps, dtype=np.int32)
            for i, token in enumerate(train_data):
                X[i] = form_alphabet.get(token['form'], 1)
            Y = np.zeros((max_steps, len(pos_alphabet) - 2), dtype=np.float32)
            for i, token in enumerate(train_data):
                Y[i, pos_alphabet.get(token['pos']) - 2] = 1.
            cost += model.train(X, Y, len(train_data))

            n_sentence += 1
            if opts.report_stops > 0 and (n_sentence % opts.report_stops == 0):
                logging.info('Processed %d sentence.' % n_sentence)
            if opts.evaluate_stops > 0 and n_sentence % opts.evaluate_stops == 0:
                logging.info('Finish %f sentences' % (float(n_sentence) / len(train_dataset)))
                p = evaluate(devel_dataset, form_alphabet, pos_alphabet, model, max_steps)
                logging.info('Devel at %d, UAS=%f' % (n_sentence, p))
                if p > best_p:
                    best_p = p
                    logging.info('New best achieved: %f' % best_p)
        logging.info('Cost %f' % cost)
        p = evaluate(devel_dataset, form_alphabet, pos_alphabet, model, max_steps)
        logging.info('Devel at the end of iteration %d, UAS=%f' % (iter, p))
        if p > best_p:
            best_p = p
            logging.info('New best achieved: %f' % best_p)
    logging.info('Finish training, best uas is %f' % best_p)


def test():
    pass

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "learn":
        learn()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        print ("usage %s [learn|test] option" % sys.argv[0])
