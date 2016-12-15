#!/usr/bin/env python
import argparse
import sys
import os
import logging
import numpy as np
from hmmlearn import hmm
from sklearn.metrics import v_measure_score
try:
    from bequick.corpus import read_conllx_dataset, get_alphabet
    from lin2015.metrics import many_to_one_score
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
    from bequick.corpus import read_conllx_dataset, get_alphabet
    from lin2015.metrics import many_to_one_score


logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
LOG = logging.getLogger('Kupiec1992')


def transform(dataset, form_alphabet, pos_alphabet):
    n_samples = sum([len(data) for data in dataset])
    n_sequences = len(dataset)
    X = np.zeros((n_samples, 1), dtype=np.int32)
    Y = np.zeros(n_samples, dtype=np.int32)
    lengths = np.zeros(n_sequences, dtype=np.int32)
    t = 0
    for i, data in enumerate(dataset):
        for token in data:
            X[t, 0] = form_alphabet.get(token['form'])
            Y[t] = pos_alphabet.get(token['pos'])
            t += 1
        lengths[i] = len(data)
    return X, Y, lengths


def main():
    flags = argparse.ArgumentParser()
    flags.add_argument("dataset", help="the path to the reference data.")
    opts = flags.parse_args()

    train_set = read_conllx_dataset(opts.dataset)
    LOG.info('loaded {0} sentence'.format(len(train_set)))
    form_alphabet = get_alphabet(train_set, 'form', init_with_default_keys=False)
    pos_alphabet = get_alphabet(train_set, 'pos', init_with_default_keys=False)
    n_pos, n_words = len(pos_alphabet), len(form_alphabet)
    LOG.info('# words: {0}, # tags: {1}'.format(n_words, n_pos))

    X, Y, lengths = transform(train_set, form_alphabet, pos_alphabet)
    LOG.info('data transformed, {0} samples, {1} sequences.'.format(X.shape[0], lengths.shape[0]))

    model = hmm.MultinomialHMM(n_components=n_pos, verbose=True)
    model.fit(X, lengths)

    Y_pred = model.predict(X, lengths)
    LOG.info("V-Measure: {0}".format(v_measure_score(Y, Y_pred)))
    LOG.info("many-to-one-Measure: {0}".format(many_to_one_score(Y, Y_pred)))

if __name__ == "__main__":
    main()
