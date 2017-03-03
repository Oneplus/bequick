#!/usr/bin/env python
from __future__ import print_function
import argparse
import logging
import numpy as np
import tensorflow as tf
from itertools import chain
from sklearn.metrics import accuracy_score
try:
    import bequick
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from bequick.alphabet import Alphabet
from bequick.tf_utils import random_uniform_matrix
from bequick.embedding import load_embedding
from bowman2015.corpus import load_json_data
tf.set_random_seed(1234)
np.random.seed(1234)
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
LOG = logging.getLogger('bowman2015')


class Model(object):
    def __init__(self, algorithm, form_size, form_dim, hidden_dim, output_dim, n_layers,
                 max_sentence1_steps, max_sentence2_steps, batch_size, regularizer):
        self.form_size = form_size
        self.form_dim = form_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.max_sentence1_steps = max_sentence1_steps
        self.max_sentence2_steps = max_sentence2_steps
        self.batch_size = batch_size

        # PLACEHOLDER
        self.X1 = tf.placeholder(tf.int32, (None, max_sentence1_steps), name="X1")
        self.L1 = tf.placeholder(tf.int32, name="L1")
        self.X2 = tf.placeholder(tf.int32, (None, max_sentence2_steps), name="X2")
        self.L2 = tf.placeholder(tf.int32, name="L2")
        self.Y = tf.placeholder(tf.int32, name='Y')

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.form_emb = tf.Variable(random_uniform_matrix(form_size, form_dim), name="form_emb")

            def get_inputs(input_placeholder, max_steps):
                # inputs for the first sentence
                inputs = tf.nn.embedding_lookup(self.form_emb, input_placeholder)
                # shape(inputs) => (batch_size, max_steps, form_dim)
                inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(inputs, max_steps, 1)]
                # shape(input_) => (batch_size, 1, form_dim)
                # after squeeze, shape(input_) => (batch_size, form_dim)
                return inputs

            inputs1 = get_inputs(self.X1, max_sentence1_steps)
            inputs2 = get_inputs(self.X2, max_sentence2_steps)

        def get_stacked_lstm_cell():
            # RNN for the 1st sentence.
            fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=True)
            bw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=True)
            fw_stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell] * n_layers, state_is_tuple=True)
            bw_stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell] * n_layers, state_is_tuple=True)
            return fw_stacked_lstm_cell, bw_stacked_lstm_cell

        # s1_fw_cell, s1_bw_cell = get_stacked_lstm_cell()
        # s2_fw_cell, s2_bw_cell = get_stacked_lstm_cell()
        fw_cell, bw_cell = get_stacked_lstm_cell()

        with tf.variable_scope('sentence1'):
            outputs1, _, _ = tf.nn.bidirectional_rnn(fw_cell, bw_cell, inputs1,
                                                     sequence_length=self.L1, dtype=tf.float32)
        with tf.variable_scope('sentence2'):
            outputs2, _, _ = tf.nn.bidirectional_rnn(fw_cell, bw_cell, inputs2,
                                                     sequence_length=self.L2, dtype=tf.float32)
        output1_bw = tf.split(1, 2, outputs1[0])[1]
        output1_fw = tf.split(1, 2, outputs1[-1])[0]
        output2_bw = tf.split(1, 2, outputs2[0])[1]
        output2_fw = tf.split(1, 2, outputs2[-1])[0]
        output = tf.concat(1, [output1_fw, output1_bw, output2_fw, output2_bw])
        # shape(output) => (32, 400)
        output = tf.nn.relu(output)
        # shape(output) => (32, 400)

        # The MLP layer
        sentence_hidden_dim = self.hidden_dim * 4
        self.W0 = tf.Variable(random_uniform_matrix(sentence_hidden_dim, sentence_hidden_dim), name="W0")
        # shape(W0) => (400, 400)
        self.b0 = tf.Variable(tf.zeros([sentence_hidden_dim]), name="b0")
        # shape(b0) => (400)
        self.W1 = tf.Variable(random_uniform_matrix(sentence_hidden_dim, self.output_dim), name="W1")
        self.b1 = tf.Variable(tf.zeros([self.output_dim]), name="b1")
        logits = tf.matmul(tf.nn.relu(tf.matmul(output, self.W0) + self.b0), self.W1) + self.b1
        self.predication = tf.nn.softmax(logits)
        # shape(pred) => (32, 3)

        # LOSS
        reg = regularizer * (tf.nn.l2_loss(self.W0) + tf.nn.l2_loss(self.b0) +
                             tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.b1))
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.Y)) + reg

        if algorithm == "adagrad":
            self.optimization = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(self.loss)
        elif algorithm == "adadelta":
            self.optimization = tf.train.AdadeltaOptimizer(learning_rate=0.01).minimize(self.loss)
        elif algorithm == "adam":
            self.optimization = tf.train.AdamOptimizer().minimize(self.loss)
        else:
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(0.01, global_step, 100000, 0.96)
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            variables = tf.trainable_variables()
            gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, variables), 5.)
            self.optimization = opt.apply_gradients(zip(gradients, variables), global_step=global_step)

    def init(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def initialize_word_embeddings(self, indices, matrix):
        self.session.run(tf.scatter_update(self.form_emb, indices, matrix))

    def train(self, x1, l1, x2, l2, y):
        """

        :param x1: array-like, (n_samples, max_sentence1_steps)
        :param l1: array-like, (n_samples,)
        :param x2: array-like, (n_samples, max_sentence2_steps)
        :param l2: array-like, (n_samples,)
        :param y: array-like, (n_samples,)
        :return:
        """
        _, cost = self.session.run([self.optimization, self.loss],
                                   feed_dict={self.X1: x1, self.L1: l1, self.X2: x2, self.L2: l2, self.Y: y})
        return cost

    def classify(self, x1, l1, x2, l2):
        """

        :param x1: array-like, (n_samples, max_sentence1_steps)
        :param l1: array-like, (n_samples,)
        :param x2: array-like, (n_samples, max_sentence2_steps)
        :param l2: array-like, (n_samples,)
        :return:
        """
        ret = self.session.run(self.predication, feed_dict={self.X1: x1, self.L1: l1, self.X2: x2, self.L2: l2})
        return ret.argmax(axis=1)


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
        max_sentence1_length = max(max_sentence1_length, len(data[1]))
        max_sentence2_length = max(max_sentence2_length, len(data[2]))
    return max_sentence1_length, max_sentence2_length


def transform(dataset, max_sentence1_steps, max_sentence2_steps, train=False, form_alphabet=None,
              label_alphabet=None):
    if form_alphabet is None:
        form_alphabet = Alphabet(use_default_initialization=True)
    if label_alphabet is None:
        label_alphabet = Alphabet(use_default_initialization=False)
    n_sentences = len(dataset)
    Y = np.zeros(n_sentences, dtype=np.int32)
    X1 = np.zeros((n_sentences, max_sentence1_steps), dtype=np.int32)
    L1 = np.zeros(n_sentences, dtype=np.int32)
    X2 = np.zeros((n_sentences, max_sentence2_steps), dtype=np.int32)
    L2 = np.zeros(n_sentences, dtype=np.int32)
    for i, data in enumerate(dataset):
        Y[i] = label_alphabet.insert(data[0]) if train else label_alphabet.get(data[0])
        l1 = len(data[1])
        X1[i, : l1] = np.array([form_alphabet.insert(t) if train else form_alphabet.get(t) for t in data[1]],
                               dtype=np.int32)
        L1[i] = l1
        l2 = len(data[2])
        X2[i, : l2] = np.array([form_alphabet.insert(t) if train else form_alphabet.get(t) for t in data[2]],
                               dtype=np.int32)
        L2[i] = l2
    return X1, L1, X2, L2, Y, form_alphabet, label_alphabet


def main():
    cmd = argparse.ArgumentParser("An implementation of Bowman et al. (2015)")
    cmd.add_argument("--form_dim", type=int, required=True, help="the dim of the form.")
    cmd.add_argument("--hidden_dim", type=int, required=True, help="the dim of the hidden output.")
    cmd.add_argument("--layers", type=int, default=1, help='the number of layers.')
    cmd.add_argument("--batch_size", type=int, default=32, help='the batch size.')
    cmd.add_argument("--algorithm", default="adam",
                     help="the algorithm [clipping_sgd, adagrad, adadelta, adam].")
    cmd.add_argument("--max_iter", default=10, type=int, help="the maximum iteration.")
    cmd.add_argument("--embedding", help="the path to the word embedding.")
    cmd.add_argument("train", default="snli_1.0_train.jsonl", help="the path to the training file.")
    cmd.add_argument("devel", default="snli_1.0_dev.jsonl", help="the path to the development file.")
    cmd.add_argument("test", default="snli_1.0_test.jsonl", help="the path to the testing file.")
    args = cmd.parse_args()

    train_data = load_json_data(args.train)
    devel_data = load_json_data(args.devel)
    test_data = load_json_data(args.test)
    LOG.info("loaded train: {0}, devel: {1}, test: {2}.".format(len(train_data), len(devel_data), len(test_data)))

    max_sentence1_steps, max_sentence2_steps = get_max_length(train_data, devel_data, test_data)
    LOG.info("max length for sentence 1: {0}, sentence 2: {1}".format(max_sentence1_steps, max_sentence2_steps))

    train_X1, train_L1, train_X2, train_L2, train_Y, form_alphabet, label_alphabet \
        = transform(train_data, max_sentence1_steps, max_sentence2_steps, True)
    devel_X1, devel_L1, devel_X2, devel_L2, devel_Y, _, _ \
        = transform(devel_data, max_sentence1_steps, max_sentence2_steps, False, form_alphabet, label_alphabet)
    test_X1, test_L1, test_X2, test_L2, test_Y, _, _ \
        = transform(test_data, max_sentence1_steps, max_sentence2_steps, False, form_alphabet, label_alphabet)

    form_size = len(form_alphabet)
    LOG.info("vocabulary size: {0}".format(form_size))
    n_classes = len(label_alphabet)
    LOG.info("number of classes: {0}".format(n_classes))

    # config and init the model
    model = Model(algorithm=args.algorithm,
                  form_size=form_size,
                  form_dim=args.form_dim,
                  hidden_dim=args.hidden_dim,
                  n_layers=args.layers,
                  output_dim=n_classes,
                  max_sentence1_steps=max_sentence1_steps,
                  max_sentence2_steps=max_sentence2_steps,
                  batch_size=args.batch_size,
                  regularizer=0.0)
    model.init()
    LOG.info("model is initialized.")

    if args.embedding is not None:
        indices, matrix = load_embedding(args.embedding, form_alphabet, args.form_dim)
        model.initialize_word_embeddings(indices, matrix)
        LOG.info("%d word embedding is loaded." % len(indices))

    n_train, n_devel, n_test = train_Y.shape[0], devel_Y.shape[0], test_Y.shape[0]
    order = np.arange(n_train, dtype=np.int32)
    for iteration in range(args.max_iter):
        np.random.shuffle(order)
        cost = 0.
        for batch_start in range(0, n_train, args.batch_size):
            batch_end = batch_start + args.batch_size if batch_start + args.batch_size < n_train else n_train
            batch = order[batch_start: batch_end]
            x1, l1, x2, l2, y = train_X1[batch], train_L1[batch], train_X2[batch], train_L2[batch], train_Y[batch]
            cost += model.train(x1, l1, x2, l2, y)
        LOG.info("cost after iteration {0}: {1}".format(iteration, cost))

        prediction = np.full(n_devel, n_classes + 1, dtype=np.int32)
        for batch_start in range(0, n_devel, args.batch_size):
            batch_end = batch_start + args.batch_size if batch_start + args.batch_size < n_devel else n_devel
            prediction[batch_start: batch_end] = model.classify(devel_X1[batch_start: batch_end],
                                                                devel_L1[batch_start: batch_end],
                                                                devel_X2[batch_start: batch_end],
                                                                devel_L2[batch_start: batch_end])
        LOG.info("dev accuracy: {0}".format(accuracy_score(devel_Y, prediction)))

        prediction = np.full(n_test, n_classes + 1, dtype=np.int32)
        for batch_start in range(0, n_test, args.batch_size):
            batch_end = batch_start + args.batch_size if batch_start + args.batch_size < n_test else n_test
            prediction[batch_start: batch_end] = model.classify(test_X1[batch_start: batch_end],
                                                                test_L1[batch_start: batch_end],
                                                                test_X2[batch_start: batch_end],
                                                                test_L2[batch_start: batch_end])
        LOG.info("test accuracy: {0}".format(accuracy_score(test_Y, prediction)))

if __name__ == "__main__":
    main()
