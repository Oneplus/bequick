#!/usr/bin/env python
import argparse
import logging
import numpy as np
from gensim.models.word2vec import Word2Vec
from hmmlearn.hmm import _BaseHMM, check_random_state
from sklearn.metrics import v_measure_score
try:
    import bequick
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from bequick.corpus import read_conllx_dataset, get_alphabet
from metrics import many_to_one_score


logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
LOG = logging.getLogger('lin2015')


class GaussianHMM(_BaseHMM):
    """

    """
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_proior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="ste", init_params="ste",
                 n_dim=100):
        _BaseHMM.__init__(self, n_components=n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_proior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params)
        self.n_dim = n_dim

    def _init(self, X, lengths=None):
        super(GaussianHMM, self)._init(X, lengths=lengths)
        self.random_state = check_random_state(self.random_state)

        if 'e' in self.init_params:
            self.mu = np.zeros((self.n_components, self.n_dim), dtype=np.float32)
            self.sigma = np.zeros((self.n_components, self.n_dim, self.n_dim), dtype=np.float32)
            # according to lin et al. (2015)
            for t in range(self.n_components):
                np.fill_diagonal(self.sigma[t], 0.45)
            self.inv_sigma = np.zeros((self.n_components, self.n_dim, self.n_dim), dtype=np.float32)
            for t in range(self.n_components):
                self.inv_sigma[t] = np.linalg.inv(self.sigma[t])

    def _check(self):
        super(GaussianHMM, self)._check()

    def _compute_log_likelihood(self, X):
        """
        Computes per-component log probability under the model.
        :param X: array-like, shape (n_samples, n_dim)
        :return: array, shape (n_samples, n_components)
        """
        n_samples = X.shape[0]
        mu = self.mu
        sigma = self.sigma
        norm_term = -0.5 * self.n_dim * np.log(2. * np.pi) - np.log(np.linalg.norm(sigma))
        inv_sigma = self.inv_sigma
        ret = np.zeros((n_samples, self.n_components), dtype=np.float32)
        for i in range(n_samples):
            for t in range(self.n_components):
                ret[i, t] = -0.5 * np.dot(np.dot((X[i] - mu[t]), inv_sigma[t]), np.transpose(X[i] - mu[t])) - norm_term
        return ret.astype(np.float64)

    def _generate_sample_from_state(self, state, random_state=None):
        raise NotImplementedError("Not implemented!")

    def _initialize_sufficient_statistics(self):
        """
        :return stats: dict
        """
        stats = super(GaussianHMM, self)._initialize_sufficient_statistics()
        stats['mu'] = np.zeros((self.n_components, self.n_dim), dtype=np.float32)
        stats['mu_norm'] = np.zeros(self.n_components, dtype=np.float32)
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob, posteriors, fwdlattice, bwdlattice):
        super(GaussianHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
        if 'e' in self.params:
            for t in range(self.n_components):
                stats['mu'][t] += np.sum(np.dot(posteriors[:, t].T, X), axis=0)
            stats['mu_norm'] += posteriors.sum(axis=0)

    def _do_mstep(self, stats):
        super(GaussianHMM, self)._do_mstep(stats)
        if 'e' in self.params:
            for t in range(self.n_components):
                self.mu[t] = stats['mu'][t] / stats['mu_norm'][t]
            stats['mu'] = np.zeros((self.n_components, self.n_dim), dtype=np.float32)
            stats['mu_norm'] = np.zeros(self.n_components, dtype=np.float32)


def transform(dataset, word2vec_model, pos_alphabet, n_dim):
    n_samples = sum([len(data) for data in dataset])
    n_sequences = len(dataset)
    X = np.zeros((n_samples, n_dim), dtype=np.int32)
    Y = np.zeros(n_samples, dtype=np.int32)
    lengths = np.zeros(n_sequences, dtype=np.int32)
    t = 0
    for i, data in enumerate(dataset):
        for token in data:
            X[t, :] = word2vec_model[token['form']]
            Y[t] = pos_alphabet.get(token['pos'])
            t += 1
        lengths[i] = len(data)
    return X, Y, lengths


def make_embedding(dataset, n_dim):
    sentences = [[token['form'] for token in data] for data in dataset]
    model = Word2Vec(sentences, size=n_dim, window=3, min_count=0, workers=1)
    return model


def main():
    flags = argparse.ArgumentParser()
    flags.add_argument("dataset", help="the path to the reference data.")
    flags.add_argument("--dim", default=20, help="the dimension of embedding.")
    opts = flags.parse_args()

    train_set = read_conllx_dataset(opts.dataset)
    LOG.info('loaded {0} sentence'.format(len(train_set)))
    form_alphabet = get_alphabet(train_set, 'form', init_with_default_keys=False)
    pos_alphabet = get_alphabet(train_set, 'pos', init_with_default_keys=False)
    n_pos, n_words = len(pos_alphabet), len(form_alphabet)
    LOG.info('# words: {0}, # tags: {1}'.format(n_words, n_pos))
    model = make_embedding(train_set, opts.dim)

    X, Y, lengths = transform(train_set, model, pos_alphabet, opts.dim)
    LOG.info('data transformed, {0} samples, {1} sequences, {2} dim.'.format(X.shape[0], lengths.shape[0], opts.dim))

    model = GaussianHMM(n_components=n_pos, random_state=1, verbose=True, n_iter=50, n_dim=opts.dim)
    model.fit(X, lengths)

    Y_pred = model.predict(X, lengths)
    LOG.info("V-Measure: {0}".format(v_measure_score(Y, Y_pred)))
    LOG.info("many-to-one-Measure: {0}".format(many_to_one_score(Y, Y_pred)))


if __name__ == "__main__":
    main()
