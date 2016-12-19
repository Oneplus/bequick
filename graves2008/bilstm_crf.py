#!/usr/bin/env python
import argparse
import logging
import numpy as np
import tensorflow as tf
try:
    import bequick
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from bequick.tf_utils import random_uniform_matrix
from bequick.corpus import read_conllx_dataset, get_alphabet
from bequick.embedding import load_embedding
try:
    from .misc import transform, get_max_steps
except ValueError:
    from misc import transform, get_max_steps

tf.set_random_seed(1234)
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
LOG = logging.getLogger('graves2008/bilstm-crf')


class Model(object):
    def __init__(self, algorithm, form_size, form_dim, hidden_dim, output_dim, n_layers, max_steps, batch_size,
                 regularizer):
        self.algorithm = algorithm
        self.form_size = form_size
        self.form_dim = form_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layer = n_layers
        self.max_steps = max_steps
        self.batch_size = batch_size

        # PLACEHOLDER: input and output
        self.x = tf.placeholder(tf.int32, (batch_size, max_steps), name="x")
        # shape(X) => (50, 32)
        self.y = tf.placeholder(tf.int32, (batch_size, max_steps), name="y")    # use the sparse version
        # shape(Y) => (50, 32)
        self.steps = tf.placeholder(tf.int32, batch_size, name="step")

        # EMBEDDING in CPU
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            self.form_emb = tf.Variable(random_uniform_matrix(form_size, form_dim), name="form_emb")
            inputs = tf.nn.embedding_lookup(self.form_emb, self.x)
            # shape(inputs) => (batch_size, max_stop, form_dim)
            inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, max_steps, inputs)]
            # shape(inputs[0]) => (batch_size, max_stop)

        # RNN
        fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=True)
        bw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=True)
        fw_stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell] * n_layers, state_is_tuple=True)
        bw_stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell] * n_layers, state_is_tuple=True)

        outputs, _, _ = tf.nn.bidirectional_rnn(fw_stacked_lstm_cell, bw_stacked_lstm_cell, inputs,
                                                sequence_length=self.steps, dtype=tf.float32)
        hidden_dim2 = self.hidden_dim * 2
        output = tf.reshape(tf.concat(1, outputs), [-1, hidden_dim2])
        self.W0 = tf.Variable(random_uniform_matrix(hidden_dim2, hidden_dim2), name="W0")
        self.b0 = tf.Variable(tf.zeros([hidden_dim2]), name="b0")
        output = tf.nn.tanh(tf.add(tf.matmul(output, self.W0), self.b0))
        self.W1 = tf.Variable(random_uniform_matrix(hidden_dim2, output_dim), name="W1")
        self.b1 = tf.Variable(tf.zeros([output_dim]), name="b1")
        output = tf.add(tf.matmul(output, self.W1), self.b1)
        self.pred = tf.reshape(output, (self.batch_size, self.max_steps, self.output_dim), name="prediction")

        # LOSS
        self.loss, self.T = tf.contrib.crf.crf_log_likelihood(self.pred, self.y, sequence_lengths=self.steps)
        self.loss = tf.reduce_mean(-self.loss)
        reg = regularizer * (tf.nn.l2_loss(self.W0) + tf.nn.l2_loss(self.b0) +
                             tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.b1) +
                             tf.nn.l2_loss(self.T))
        self.loss += reg

        if algorithm == "adam":
            self.optm = tf.train.AdamOptimizer().minimize(self.loss)
        elif algorithm == "adagrad":
            self.optm = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(self.loss)
        elif algorithm == "adadelta":
            self.optm = tf.train.AdadeltaOptimizer(learning_rate=0.01).minimize(self.loss)
        else:
            # GRADIENTS AND CLIPPING
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(0.01, global_step, 100000, 0.96)
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5.)
            self.optm = opt.apply_gradients(zip(grads, tvars), global_step=global_step)

    def initialize_word_embeddings(self, session, indices, matrix):
        _indices = [tf.to_int32(i) for i in indices]
        session.run(tf.scatter_update(self.form_emb, _indices, matrix))

    def train(self, session, x, y, steps):
        _, cost = session.run([self.optm, self.loss], feed_dict={self.x: x, self.y: y, self.steps: steps})
        return cost

    def decode(self, session, x, steps):
        return session.run([self.pred, self.T], feed_dict={self.x: x, self.steps: steps})


def evaluate(dataset, session, model):
    """

    :param dataset: tuple(numpy.array, tuple.numpy.array)
    :param session:
    :param model:
    :return: float, the precision
    """
    n_corr, n_total = 0, 0
    l = dataset[0].shape[0]
    batch_size = model.batch_size
    xs, ys, steps = dataset
    for n in range(0, l, batch_size):
        end = min(n + batch_size, l)
        if end - n < batch_size:
            x = np.zeros((batch_size, xs.shape[1]), dtype=np.int32)
            y = np.zeros((batch_size, ys.shape[1]), dtype=np.int32)
            step = np.zeros(batch_size, dtype=np.int32)
            x[: end - n, :] = xs[n: end, :]
            y[: end - n, :] = ys[n: end, :]
            step[: end - n] = steps[n: end]
        else:
            x, y, step = xs[n: end, :], ys[n: end, :], steps[n: end]
        y_pred, T = model.decode(session, x, step)
        for i in range(end - n):
            if step[i] == 0:
                continue
            pred, _ = tf.contrib.crf.viterbi_decode(y_pred[i, : step[i]], T)
            n_total += step[i]
            n_corr += np.array(y[i, : step[i]] == pred, dtype=np.int32).sum()
    return float(n_corr) / n_total


def main():
    usage = "A implementation of the Bidirectional LSTM CRF."
    cmd = argparse.ArgumentParser(usage=usage)
    cmd.add_argument("--embedding", help="The path to the embedding file.")
    cmd.add_argument("--max-iter", dest="max_iter", type=int, default=50, help="The number of max iteration.")
    cmd.add_argument("--layers", dest="layers", type=int, default=1, help="The size of layers.")
    cmd.add_argument("--hidden-size", dest="hidden_size", type=int, default=100, help="The size of hidden layer.")
    cmd.add_argument("--embedding-size", dest="embedding_size", type=int, default=100, help="The size of embedding.")
    cmd.add_argument("--evaluate-stops", dest="evaluate_stops", type=int, default=-1, help="evaluate each n sentence.")
    cmd.add_argument("--report-stops", dest="report_stops", type=int, default=-1, help="evaluate each n sentence.")
    cmd.add_argument("--lambda", dest="lambda_", type=float, default=1e-8, help="The regularizer parameter.")
    cmd.add_argument("--dropout", dest="dropout", type=float, default=0.5, help="The probability for dropout.")
    cmd.add_argument("--algorithm", default="adam", help="The learning algorithm [adam, adagrad, clipped_sgd].")
    cmd.add_argument("--batch-size", dest="batch_size", type=int, default=32, help="The batch size.")
    cmd.add_argument("reference", help="The path to the reference file.")
    cmd.add_argument("development", help="The path to the development file.")
    cmd.add_argument("test", help="The path to the test file.")
    opts = cmd.parse_args()

    train_set = read_conllx_dataset(opts.reference)
    devel_set = read_conllx_dataset(opts.development)
    test_set = read_conllx_dataset(opts.test)
    form_alphabet = get_alphabet(train_set, 'form')
    pos_alphabet = get_alphabet(train_set, 'pos')
    LOG.info("# train: {0}, # devel: {1}, # test: {2}".format(len(train_set), len(devel_set), len(test_set)))
    LOG.info("# form alpha: {0}, w/ 0=placeholder, 1=UNK".format(len(form_alphabet)))
    LOG.info("# pos alpha: {0}, w/ 0=placeholder, 1=UNK".format(len(pos_alphabet)))
    max_steps = get_max_steps(train_set, devel_set, test_set)
    LOG.info('max steps {0}'.format(max_steps))

    train_set = transform(train_set, form_alphabet, pos_alphabet, max_steps)
    devel_set = transform(devel_set, form_alphabet, pos_alphabet, max_steps)
    test_set = transform(test_set, form_alphabet, pos_alphabet, max_steps)
    LOG.info("data is transformed.")

    model = Model(algorithm=opts.algorithm,
                  form_size=len(form_alphabet),
                  form_dim=opts.embedding_size,
                  hidden_dim=opts.hidden_size,
                  output_dim=len(pos_alphabet),   # should also counting the BAD0, because
                  n_layers=opts.layers,
                  max_steps=max_steps,
                  batch_size=opts.batch_size,
                  regularizer=opts.lambda_)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    if opts.embedding is not None:
        indices, matrix = load_embedding(opts.embedding, form_alphabet, opts.embedding_size, True)
        model.initialize_word_embeddings(session, indices, matrix)
        LOG.info('embedding is loaded.')

    n_sentence = 0
    best_p, test_score = 0., 0.

    l = train_set[0].shape[0]
    xs, ys, steps = train_set
    mask = list(range(l))
    for iteration in range(1, opts.max_iter + 1):
        LOG.info('iteration {0}'.format(iteration))
        np.random.shuffle(mask)
        cost = 0.
        for n in range(0, l, opts.batch_size):
            end = min(n + opts.batch_size, l)
            if end - n < opts.batch_size:
                x = np.zeros((opts.batch_size, max_steps), dtype=np.int32)
                y = np.zeros((opts.batch_size, max_steps), dtype=np.int32)
                step = np.zeros(opts.batch_size, dtype=np.int32)
                x[: end - n, :] = xs[mask[n: end], :]
                y[: end - n, :] = ys[mask[n: end], :]
                step[: end - n] = steps[mask[n: end]]
            else:
                x, y, step = xs[mask[n: end], :], ys[mask[n: end], :], steps[mask[n: end]]
            cost += model.train(session, x, y, step)
            n_sentence += 1
            if opts.report_stops > 0 and (n_sentence % opts.report_stops == 0):
                LOG.info('Processed {0} sentence.'.format(n_sentence))
            if opts.evaluate_stops > 0 and n_sentence % opts.evaluate_stops == 0:
                LOG.info('Finish {0} sentences'.format(float(n_sentence) / len(train_set)))
                p = evaluate(devel_set, session, model)
                LOG.info('Development score is: {0}'.format(p))
                if p > best_p:
                    best_p = p
                    test_score = evaluate(test_set, session, model)
                    LOG.info('New best is achieved, Test score is: {0}'.format(test_score))

        p = evaluate(devel_set,  session, model)
        LOG.info('At the end of iteration {0}, cost={1}, accuracy={2}'.format(iteration, cost, p))
        if p > best_p:
            best_p = p
            test_score = evaluate(test_set, session, model)
            LOG.info('New best is achieved, Test score is: {0}'.format(test_score))

    LOG.info('Finish training, best development accuracy is {0}, test accuracy is {1}'.format(best_p, test_score))


if __name__ == "__main__":
    main()
