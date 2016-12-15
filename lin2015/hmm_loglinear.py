#!/usr/bin/env python
import argparse
import sys
import os
import logging
import numpy as np
import tensorflow as tf
from hmmlearn.hmm import _BaseHMM, check_random_state
from hmmlearn.utils import iter_from_X_lengths
from sklearn import metrics
try:
    from bequick.corpus import read_conllx_dataset, get_alphabet
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
    from bequick.corpus import read_conllx_dataset, get_alphabet


logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
LOG = logging.getLogger('berg-kirkpatrick2010')


class LogLinear(object):
    """ The tensorflow logistic regression model. """
    def __init__(self, n_components, n_templates, n_features, n_iter=10, batch_size=32, tol=1e-2, lambda_=0.3):
        self.n_components = n_components
        self.n_templates = n_templates
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.tol = tol

        self.X = tf.placeholder(dtype=tf.int64, shape=(None, n_templates), name="X")  # (None, n_templates)
        self.Y = tf.placeholder(dtype=tf.float32, shape=(None, n_components), name="Y")   # (None, n_components)
        # NOTE: (n_features, n_components)
        self.W = tf.Variable(tf.random_normal((n_features, n_components), stddev=0.01), dtype=tf.float32)
        self.b = tf.Variable(tf.zeros(n_components), dtype=tf.float32)
        self.W2 = tf.nn.embedding_lookup(self.W, self.X)    # (None, n_templates, n_components)
        self.W2 = tf.reshape(tf.reduce_sum(self.W2, axis=1), shape=(-1, n_components))   # (None, 1, n_components)
        logits = tf.add(self.W2, self.b)
        self.logprob_op = tf.nn.log_softmax(logits)
        regularizer = lambda_ * (tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b))
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output, self.Y)) + regularizer
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.Y))

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.init_op = tf.global_variables_initializer()

        self.session = tf.Session()
        self.session.run(self.init_op)

    def fit(self, X, Y):
        """

        :param X: array-like, (n_samples, n_features)
        :param Y: array-like, (n_samples, n_components)
        :return cost: float
        """
        history = [0., 0.]
        for i in range(self.n_iter):
            n_samples = X.shape[0]
            cost = 0.
            for batch_start in range(0, n_samples, self.batch_size):
                batch_end = (batch_start + self.batch_size if batch_start + self.batch_size < n_samples else n_samples)
                _, batch_cost = self.session.run([self.train_op, self.loss], feed_dict={
                    self.X: X[batch_start: batch_end],
                    self.Y: Y[batch_start: batch_end]
                })
                cost += batch_cost
            LOG.info("Train LR, iter={0}, cost={1}".format(i, cost))
            history[0], history[1] = history[1], cost
            if np.abs(history[1] - history[0]) < self.tol:
                break

    def predict_log_proba(self, X):
        """

        :param X: array-like, (n_samples, n_features)
        :return log_prob: array-like, (n_samples, n_components)
        """
        return self.session.run(self.logprob_op, feed_dict={self.X: X})


class LogLinearHMM(_BaseHMM):
    """

    """
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_proior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="ste", init_params="ste"):
        _BaseHMM.__init__(self, n_components=n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_proior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params)

    def _init(self, X, lengths=None):
        if not self._check_input_symbols(X):
            raise ValueError("expected a sample from a Multinomial distribution.")

        super(LogLinearHMM, self)._init(X, lengths=lengths)
        self.random_state = check_random_state(self.random_state)

        if 'e' in self.init_params:
            if not hasattr(self, "n_features"):
                symbols = set()
                for i, j in iter_from_X_lengths(X, lengths):
                    symbols |= set(X[i:j].flatten())
                self.n_features = len(symbols)
            if not hasattr(self, "n_instances"):
                self.n_instances = X.shape[0]
            if not hasattr(self, "n_templates"):
                self.n_templates = X[0].shape[0]
            self.logres_model = LogLinear(n_components=self.n_components, n_templates=self.n_templates,
                                          n_features=self.n_features, batch_size=128)

    def _check(self):
        super(LogLinearHMM, self)._check()

    def _compute_log_likelihood(self, X):
        """
        Computes per-component log probability under the model.
        :param X: array-like, shape (n_samples, n_features)
        :return: array, shape (n_samples, n_components)
        """
        return self.logres_model.predict_log_proba(X).astype(np.float64)

    def _generate_sample_from_state(self, state, random_state=None):
        raise NotImplementedError("Not implemented!")

    def _initialize_sufficient_statistics(self):
        stats = super(LogLinearHMM, self)._initialize_sufficient_statistics()
        stats['ctx'] = np.zeros(shape=(self.n_instances, self.n_templates), dtype=np.float32)   # the context
        stats['posteriors'] = np.zeros(shape=(self.n_instances, self.n_components), dtype=np.float32)
        stats['ntokens'] = 0
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob, posteriors, fwdlattice, bwdlattice):
        super(LogLinearHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
        if 'e' in self.params:
            for t, symbol in enumerate(X):
                n = stats['ntokens']
                stats['ctx'][n] = symbol
                stats['posteriors'][n] = posteriors[t]
                stats['ntokens'] += 1

    def _do_mstep(self, stats):
        super(LogLinearHMM, self)._do_mstep(stats)
        if 'e' in self.params:
            self.logres_model.fit(stats['ctx'], stats['posteriors'])
            stats['ntokens'] = 0

    def _check_input_symbols(self, X):
        """Check if ``X`` is a sample from a Multinomial distribution.
        That is ``X`` should be an array of non-negative integers from
        range ``[min(X), max(X)]``, such that each integer from the range
        occurs in ``X`` at least once.
        For example ``[0, 0, 2, 1, 3, 1, 1]`` is a valid sample from a
        Multinomial distribution, while ``[0, 0, 3, 5, 10]`` is not.
        """
        symbols = np.concatenate(X)
        if (len(symbols) == 1 or  # not enough data
                    symbols.dtype.kind != 'i' or  # not an integer
                (symbols < 0).any()):  # contains negative integers
            return False

        symbols.sort()
        return np.all(np.diff(symbols) <= 1)


def transform(dataset, pos_alphabet, feature_alphabet=None):
    if feature_alphabet is None:
        feature_alphabet = {}

    n_samples = sum([len(data) for data in dataset])
    n_sequences = len(dataset)
    n_features = 9
    X = np.zeros((n_samples, n_features), dtype=np.int32)
    Y = np.zeros(n_samples, dtype=np.int32)
    lengths = np.zeros(n_sequences, dtype=np.int32)
    t = 0
    for i, data in enumerate(dataset):
        for j, token in enumerate(data):
            w_minus_2 = data[j - 2]['form'] if j >= 2 else '_bos_'
            w_minus_1 = data[j - 1]['form'] if j >= 1 else '_bos_'
            w_0 = token['form']
            w_plus_1 = data[j + 1]['form'] if j + 1 < len(data) else '_eos_'
            w_plus_2 = data[j + 2]['form'] if j + 2 < len(data) else '_eos_'
            features = [
                'w[-2]={0}'.format(w_minus_2),
                'w[-1]={0}'.format(w_minus_1),
                'w[0]={0}'.format(w_0),
                'w[+1]={0}'.format(w_plus_1),
                'w[+2]={0}'.format(w_plus_2),
                'w[-2,-1]={0}#{1}'.format(w_minus_2, w_minus_1),
                'w[-1,0]={0}#{1}'.format(w_minus_1, w_0),
                'w[0,+1]={0}#{1}'.format(w_0, w_plus_1),
                'w[+1,+2]={0}#{1}'.format(w_plus_1, w_plus_2),
            ]
            for feature in features:
                if feature not in feature_alphabet:
                    feature_alphabet[feature] = len(feature_alphabet)
            X[t, :] = np.array([feature_alphabet.get(feature) for feature in features], dtype=np.int32)
            Y[t] = pos_alphabet.get(token['pos'])
            t += 1
        lengths[i] = len(data)
    return X, Y, lengths, feature_alphabet


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

    X, Y, lengths, feature_alphabet = transform(train_set, pos_alphabet)
    LOG.info('data transformed, {0} samples, {1} sequences {2} features.'.format(
        X.shape[0], lengths.shape[0], len(feature_alphabet)))

    model = LogLinearHMM(n_components=n_pos, verbose=True, n_iter=50)
    model.fit(X, lengths)

    Y_pred = model.predict(X, lengths)
    LOG.info("V-Measure: {0}".format(v_measure_score(Y, Y_pred)))
    LOG.info("many-to-one-Measure: {0}".format(many_to_one_score(Y, Y_pred)))

if __name__ == "__main__":
    main()
