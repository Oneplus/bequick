#!/usr/bin/env python
import math
import tensorflow as tf


class Model(object):
    def __init__(self,
                 form_size,
                 form_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 max_steps):
        self.form_size = form_size
        self.form_dim = form_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layer = n_layers

        # input / output, by default, the only one instance is input.
        self.X = tf.placeholder(tf.int32, [max_steps,], name="X")
        self.Y = tf.placeholder(tf.float32, [max_steps, output_dim], name="Y")
        self.early_stop = tf.placeholder(tf.int32, [1], name="len")

        width = 6. / math.sqrt(form_size + form_dim)
        self.form_emb = tf.Variable(
            tf.random_uniform([self.form_size, self.form_dim], -width, width),
            name="form_emb")

        inputs = tf.nn.embedding_lookup(self.form_emb, self.X)
        inputs = tf.reshape(inputs, [1, max_steps, self.form_dim])
        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, max_steps, inputs)]

        # RNN
        lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=True)
        self.stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * n_layers, state_is_tuple=True)

        self.initial_state = self.stacked_lstm.zero_state(1, tf.float32)

        outputs, state = tf.nn.rnn(self.stacked_lstm,
                inputs,
                initial_state=self.initial_state,
                sequence_length=self.early_stop)

        output = tf.reshape(tf.concat(1, outputs), [-1, self.hidden_dim])

        width = 6. / math.sqrt(self.hidden_dim + self.output_dim)
        self.W0 = tf.Variable(
            tf.random_uniform([self.hidden_dim, self.output_dim], -width, width),
            name="W0")
        self.b0 = tf.Variable(tf.zeros([self.output_dim]), name="b0")
        self.possibility = tf.nn.softmax(tf.matmul(output, self.W0) + self.b0)

        regularizer = 1e-5 * (tf.nn.l2_loss(self.W0) + tf.nn.l2_loss(self.b0))
        self.loss = (tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.possibility, self.Y) + regularizer))
        self.optm = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss)

    def init(self):
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def train(self, X, Y, steps):
        _, cost = self.session.run(
                [self.optm, self.loss],
                feed_dict = {
                    self.early_stop: [steps],
                    self.X: X,
                    self.Y: Y
                    })
        return cost

    def classify(self, X, steps):
        return self.session.run(self.possibility,
                feed_dict = {
                    self.early_stop: [steps],
                    self.X: X
                    })
