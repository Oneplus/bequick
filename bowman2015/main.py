#!/usr/bin/env python
import argparse
import random
import logging
import numpy as np
from collections import namedtuple
from itertools import chain
from bowman2015.utils import batch
from bowman2015.model import Model
random.seed(1234)

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
Instance = namedtuple('Instance', ['gold_label', 'sentence1', 'sentence2'], verbose=False)


def get_vocabulary_size(train_data, devel_data, test_data):
    """
    Get the size of vocabulary, assuming the index starting from 0
    :param train_data:
    :param devel_data:
    :param test_data:
    :return:
    """
    ret = 0
    for data in chain(train_data, devel_data, test_data):
        for i in chain(data.sentence1, data.sentence2):
            ret = max(ret, i)
    return ret + 1


def get_number_of_classes(train_data, devel_data, test_data):
    """
    Get the size of classes, assuming the index starting from 0
    :param train_data:
    :param devel_data:
    :param test_data:
    :return:
    """
    return max(data.gold_label for data in chain(train_data, devel_data, test_data)) + 1


def get_max_length(train_data, devel_data, test_data):
    """
    Get the maximum length of the first sentence and second sentence from the dataset.
    :param train_data:
    :param devel_data:
    :param test_data:
    :return:
    """
    max_sentence1_length, max_sentence2_length = 0, 0
    for data in chain(train_data, devel_data, test_data):
        max_sentence1_length = max(max_sentence1_length, len(data.sentence1))
        max_sentence2_length = max(max_sentence2_length, len(data.sentence2))
    return max_sentence1_length, max_sentence2_length


def train(train_data, devel_data, test_data, model, max_iteration=10, batch_size=32):
    """

    :param train_data:
    :param devel_data:
    :param test_data:
    :param model: bowman2015.Model
    :param max_iteration:
    :param batch_size:
    :return:
    """
    def transduce(chunk):
        Y = np.zeros(shape=(batch_size, model.output_dim), dtype=np.float32)
        X1 = np.zeros(shape=(model.max_sentence1_steps, batch_size), dtype=np.int32)
        X2 = np.zeros(shape=(model.max_sentence2_steps, batch_size), dtype=np.int32)
        for b, c in enumerate(chunk):
            Y[b, c.gold_label] = 1.
            X1[: len(c.sentence1), b] = c.sentence1
            X2[: len(c.sentence2), b] = c.sentence2
        return X1, X2, Y

    for iteration in range(max_iteration):
        random.shuffle(train_data)
        cost = 0.
        for chunk in batch(train_data, batch_size):
            X1, X2, Y = transduce(chunk)
            cost += model.train(X1, X2, Y)
        logging.info("cost after iteration %d: %f" % (iteration, cost))

        n_corr, n_total = 0, 0
        for chunk in batch(devel_data, batch_size):
            X1, X2, Y = transduce(chunk)
            prediction = model.classify(X1, X2)
            n_total += len(chunk)
            n_corr += sum((1 if np.argmax(p) == np.argmax(y) else 0) for p, y in zip(prediction, Y))
        logging.info("precision %f" % (float(n_corr) / n_total))

        n_corr, n_total = 0, 0
        for chunk in batch(test_data, batch_size):
            X1, X2, Y = transduce(chunk)
            prediction = model.classify(X1, X2)
            n_total += len(chunk)
            n_corr += sum((1 if np.argmax(p) == np.argmax(y) else 0) for p, y in zip(prediction, Y))
        logging.info("precision %f" % (float(n_corr) / n_total))


def load_data(path):
    """

    :param path: str, the path to the data file.
    :return: list(list(int))
    """
    ret = []
    for line in open(path, 'r'):
        tokens = line.strip().split('\t')
        ret.append(Instance(int(tokens[0]), [int(i) for i in tokens[1].split()], [int(i) for i in tokens[2].split()]))
    return ret


def main():
    usage = "An implementation of A large annotated corpus for learning natural language inference"
    cmd = argparse.ArgumentParser(usage=usage)
    cmd.add_argument("--form_dim", type=int, required=True, help="the dim of the form.")
    cmd.add_argument("--hidden_dim", type=int, required=True, help="the dim of the hidden output.")
    cmd.add_argument("--layers", type=int, default=1, help='the number of layers.')
    cmd.add_argument("--batch_size", type=int, default=32, help='the batch size.')
    cmd.add_argument("train", help="the path to the training file.")
    cmd.add_argument("devel", help="the path to the development file.")
    cmd.add_argument("test", help="the path to the testing file.")

    args = cmd.parse_args()

    train_data = load_data(args.train)
    logging.info("training data is loaded with %d instances." % len(train_data))

    devel_data = load_data(args.devel)
    logging.info("development data is loaded with %d instances." % len(devel_data))

    test_data = load_data(args.test)
    logging.info("development data is loaded with %d instances." % len(test_data))

    max_sentence1_length, max_sentence2_length = get_max_length(train_data, devel_data, test_data)
    logging.info("max length for sentence 1: %d" % max_sentence1_length)
    logging.info("max length for sentence 2: %d" % max_sentence2_length)

    form_size = get_vocabulary_size(train_data, devel_data, test_data)
    logging.info("vocabulary size: %d" % form_size)

    num_classes = get_number_of_classes(train_data, devel_data, test_data)
    logging.info("number of classes: %d" % num_classes)

    model = Model(form_size=form_size,
                  form_dim=args.form_dim,
                  hidden_dim=args.hidden_dim,
                  n_layers=args.layers,
                  output_dim=num_classes,
                  max_sentence1_steps=max_sentence1_length,
                  max_sentence2_steps=max_sentence2_length,
                  batch_size=args.batch_size)
    model.init()
    train(train_data, devel_data, test_data, model)


if __name__ == "__main__":
    main()
