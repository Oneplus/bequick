#!/usr/bin/env python
import math
import tensorflow as tf


class Model(object):
    def __init__(self,
                 form_size,
                 form_dim,
                 hidden_dim,
                 output_dim,
                 n_layers):
        self.form_size = form_size
        self.form_dim = form_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layer = n_layers

        width = 6. / math.sqrt(form_size + form_dim)
        self.form_emb = tf.Variable(
            tf.random_uniform([self.form_size, self.form_dim], -width, width),
            name="form_emb")
        lstm = tf.nn.rnn_cell.BasicLSTMCell(form_size)
        self.stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * n_layers)

        width = 6. / math.sqrt(self.hidden_dim + self.output_dim)
        self.W0 = tf.Variable(
            tf.random_uniform([self.hidden_dim, self.output_dim], -width, width),
            name="W0")
        self.b0 = tf.Variable(tf.zeros([self.output_dim]), name="b0")

    def init(self):
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def train(self, X, Y):
        initial_state = state = self.stacked_lstm.zero_state(1, tf.float32)
        loss = 0.
        for x, y in zip(X, Y):
            output, state = self.stacked_lstm(x, state)
            logits = tf.matmult(output, self.W0) + self.b0
            possibility = tf.nn.softmax(logits)
            loss += tf.nn.softmax_cross_entropy_with_logits(possibility, y)
        self.optm = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)
        return loss

    def classify(self, X):
        return []