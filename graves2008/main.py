#!/usr/bin/env python
import sys
import argparse
import logging
import random
import numpy as np
from corpus import read_dataset, get_alphabet
from model import Model

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


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
    conf = argparse.ArgumentParser("Learning component for chen and manning (2014)'s parser")
    conf.add_argument("--model", help="The path to the model.")
    conf.add_argument("--embedding", help="The path to the embedding file.")
    conf.add_argument("--reference", help="The path to the reference file.")
    conf.add_argument("--development", help="The path to the development file.")
    conf.add_argument("--max-iter", dest="max_iter",type=int, default=10,
                      help="The number of max iteration.")
    conf.add_argument("--layers", dest="layers", type=int, default=2, help="The size of layers.")
    conf.add_argument("--hidden-size", dest="hidden_size", type=int, default=200, help="The size of hidden layer.")
    conf.add_argument("--embedding-size", dest="embedding_size", type=int, default=100, help="The size of embedding.")
    conf.add_argument("--evaluate-stops", dest="evaluate_stops", type=int, default=-1, help="evaluate each n sentence.")
    conf.add_argument("--lambda", dest="lambda", type=float, default=1e-8, help="The regularizer parameter.")
    conf.add_argument("--dropout", dest="dropout", type=float, default=0.5, help="The probability for dropout.")
    opts = conf.parse_args(sys.argv[2:])

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

    model = Model(
        form_size=len(form_alphabet),
        form_dim=opts.embedding_size,
        hidden_dim=opts.hidden_size,
        output_dim=(len(pos_alphabet) - 2),
        n_layers=opts.layers,
        max_steps=max_steps
    )
    model.init()

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
