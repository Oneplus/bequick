#!/usr/bin/env python
import tensorflow as tf
try:
    import bequick
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from bequick.tf_utils import random_uniform_matrix
try:
    from .tb_parser import Parser
except (ValueError, SystemError) as e:
    from tb_parser import Parser

tf.set_random_seed(1234)


def initialize_word_embeddings(session, form_emb, indices, matrix):
    _indices = [tf.to_int32(i) for i in indices]
    session.run(tf.scatter_update(form_emb, _indices, matrix))


def unpack_inputs(inputs):
    form = [_[0] for _ in inputs]
    pos = [_[1] for _ in inputs]
    deprel = [_[2] for _ in inputs]
    return form, pos, deprel


class Network(object):
    def __init__(self, form_size, form_dim, pos_size, pos_dim, deprel_size, deprel_dim, hidden_dim, output_dim,
                 dropout, l2):
        self.form_size = form_size
        self.form_dim = form_dim
        self.pos_size = pos_size
        self.pos_dim = pos_dim
        self.deprel_size = deprel_size
        self.deprel_dim = deprel_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # DIMENSIONS
        self.form_features_dim = len(Parser.FORM_NAMES) * self.form_dim
        self.pos_features_dim = len(Parser.POS_NAMES) * self.pos_dim
        self.deprel_features_dim = len(Parser.DEPREL_NAMES) * self.deprel_dim
        self.input_dim = self.form_features_dim + self.pos_features_dim + self.deprel_features_dim

        # CONFIG
        self.dropout = dropout
        self.l2 = l2

        # INPUT
        self.form = tf.placeholder(tf.int32, shape=(None, len(Parser.FORM_NAMES), ), name="form_i")
        self.pos = tf.placeholder(tf.int32, shape=(None, len(Parser.POS_NAMES), ), name="pos_i")
        self.deprel = tf.placeholder(tf.int32, shape=(None, len(Parser.DEPREL_NAMES), ), name="deprel_i")


class Classifier(Network):
    def __init__(self, form_size, form_dim, pos_size, pos_dim, deprel_size, deprel_dim, hidden_dim, output_dim,
                 dropout, l2):
        super(Classifier, self).__init__(form_size, form_dim, pos_size, pos_dim, deprel_size, deprel_dim, hidden_dim,
                                         output_dim, dropout, l2)
        self.output = tf.placeholder(tf.int32, shape=(None, ), name="y_o")

        # EMBEDDING in CPU
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            self.form_emb = tf.Variable(random_uniform_matrix(self.form_size, self.form_dim), name="form_emb")
            self.pos_emb = tf.Variable(random_uniform_matrix(self.pos_size, self.pos_dim), name="pos_emb")
            self.deprel_emb = tf.Variable(random_uniform_matrix(self.deprel_size, self.deprel_dim), name="deprel_emb")
            inputs = tf.concat(1, [
                tf.reshape(tf.nn.embedding_lookup(self.form_emb, self.form), [-1, self.form_features_dim]),
                tf.reshape(tf.nn.embedding_lookup(self.pos_emb, self.pos), [-1, self.pos_features_dim]),
                tf.reshape(tf.nn.embedding_lookup(self.deprel_emb, self.deprel), [-1, self.deprel_features_dim])
            ])

        # PARAMS
        self.W0 = tf.Variable(random_uniform_matrix(self.input_dim, self.hidden_dim), name="W0")
        self.b0 = tf.Variable(tf.zeros([self.hidden_dim]), name="b0")
        self.W1 = tf.Variable(random_uniform_matrix(self.hidden_dim, self.output_dim), name="W1")
        self.b1 = tf.Variable(tf.zeros([self.output_dim]), name="b1")

        # PREDICTION
        hidden_layer = tf.nn.relu(tf.add(tf.matmul(inputs, self.W0), self.b0))
        self.prediction = tf.add(tf.matmul(hidden_layer, self.W1), self.b1)

        # LOSS
        regularizer = tf.nn.l2_loss(self.W0) + tf.nn.l2_loss(self.b0) + tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.b1)
        if self.dropout > 0:
            hidden_layer2 = tf.nn.dropout(hidden_layer, self.dropout)
        else:
            hidden_layer2 = hidden_layer
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.add(tf.matmul(hidden_layer2, self.W1), self.b1), self.output)) + l2 * regularizer
        self.optimization = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(self.loss)

    def train(self, session, inputs, outputs):
        form, pos, deprel = inputs
        _, cost = session.run([self.optimization, self.loss],
                              feed_dict={self.form: form, self.pos: pos, self.deprel: deprel, self.output: outputs})
        return cost

    def classify(self, session, inputs):
        form, pos, deprel = inputs
        prediction = session.run(self.prediction, feed_dict={self.form: form, self.pos: pos, self.deprel: deprel})
        return prediction


class DeepQNetwork(Network):
    def __init__(self, form_size, form_dim, pos_size, pos_dim, deprel_size, deprel_dim, hidden_dim, output_dim, dropout,
                 l2):
        super(DeepQNetwork, self).__init__(form_size, form_dim, pos_size, pos_dim, deprel_size, deprel_dim, hidden_dim,
                                           output_dim, dropout, l2)
        self.output = tf.placeholder(tf.float32, shape=(None, ), name="y_o")

        # EMBEDDING in CPU
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            self.form_emb = tf.Variable(random_uniform_matrix(self.form_size, self.form_dim), name="form_emb")
            self.pos_emb = tf.Variable(random_uniform_matrix(self.pos_size, self.pos_dim), name="pos_emb")
            self.deprel_emb = tf.Variable(random_uniform_matrix(self.deprel_size, self.deprel_dim), name="deprel_emb")
            inputs = tf.concat(1, [
                tf.reshape(tf.nn.embedding_lookup(self.form_emb, self.form), [-1, self.form_features_dim]),
                tf.reshape(tf.nn.embedding_lookup(self.pos_emb, self.pos), [-1, self.pos_features_dim]),
                tf.reshape(tf.nn.embedding_lookup(self.deprel_emb, self.deprel), [-1, self.deprel_features_dim])
            ])

            # according to minh et al (2015), target network is initialized as zero
            self.tgt_form_emb = tf.Variable(tf.zeros((self.form_size, self.form_dim)), name="tgt_form_emb")
            self.tgt_pos_emb = tf.Variable(tf.zeros((self.pos_size, self.pos_dim)), name="tgt_pos_emb")
            self.tgt_deprel_emb = tf.Variable(tf.zeros((self.deprel_size, self.deprel_dim)), name="tgt_deprel_emb")
            tgt_inputs = tf.concat(1, [
                tf.reshape(tf.nn.embedding_lookup(self.tgt_form_emb, self.form), [-1, self.form_features_dim]),
                tf.reshape(tf.nn.embedding_lookup(self.tgt_pos_emb, self.pos), [-1, self.pos_features_dim]),
                tf.reshape(tf.nn.embedding_lookup(self.tgt_deprel_emb, self.deprel), [-1, self.deprel_features_dim])
            ])

        # PARAMS
        self.W0 = tf.Variable(random_uniform_matrix(self.input_dim, self.hidden_dim), name="W0")
        self.b0 = tf.Variable(tf.zeros([self.hidden_dim]), name="b0")
        self.W1 = tf.Variable(random_uniform_matrix(self.hidden_dim, self.output_dim), name="W1")
        self.b1 = tf.Variable(tf.zeros([self.output_dim]), name="b1")

        self.tgt_W0 = tf.Variable(tf.zeros((self.input_dim, self.hidden_dim)), name="tgt_W0")
        self.tgt_b0 = tf.Variable(tf.zeros([self.hidden_dim]), name="tgt_b0")
        self.tgt_W1 = tf.Variable(tf.zeros((self.hidden_dim, self.output_dim)), name="tgt_W1")
        self.tgt_b1 = tf.Variable(tf.zeros([self.output_dim]), name="tgt_b1")

        # PARAM SYNC
        self.update_op = [
            self.tgt_form_emb.assign(self.form_emb),
            self.tgt_pos_emb.assign(self.pos_emb),
            self.tgt_deprel_emb.assign(self.deprel_emb),
            self.tgt_W0.assign(self.W0),
            self.tgt_b0.assign(self.b0),
            self.tgt_W1.assign(self.W1),
            self.tgt_b1.assign(self.b1)
        ]

        # MLP
        hidden_layer = tf.nn.relu(tf.add(tf.matmul(inputs, self.W0), self.b0))
        self.q_function = tf.add(tf.matmul(hidden_layer, self.W1), self.b1)

        tgt_hidden_layer = tf.nn.relu(tf.add(tf.matmul(tgt_inputs, self.tgt_W0), self.tgt_b0))
        self.tgt_q_function = tf.add(tf.matmul(tgt_hidden_layer, self.tgt_W1), self.tgt_b1)

        # LOSS
        if self.dropout > 0:
            hidden_layer2 = tf.nn.dropout(hidden_layer, self.dropout)
        else:
            hidden_layer2 = hidden_layer
        self.action = tf.placeholder(tf.int32, shape=(None, ), name="action")
        actions_one_hot = tf.one_hot(self.action, self.output_dim, 1.0, 0.0, name='action_one_hot')
        predicted_q = tf.reduce_sum(tf.add(tf.matmul(hidden_layer2, self.W1), self.b1) * actions_one_hot,
                                    reduction_indices=1)
        regularizer = tf.nn.l2_loss(self.W0) + tf.nn.l2_loss(self.b0) + tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.b1)

        self.loss = tf.reduce_mean(tf.square(predicted_q - self.output)) + l2 * regularizer
        # self.optimization = tf.train.RMSPropOptimizer(learning_rate=0.00025, momentum=0.95).minimize(self.loss)
        self.optimization = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self, session, inputs, action, outputs):
        form, pos, deprel = inputs
        _, cost = session.run([self.optimization, self.loss], feed_dict={
            self.form: form, self.pos: pos, self.deprel: deprel, self.action: action, self.output: outputs})
        return cost

    def policy(self, session, inputs):
        form, pos, deprel = inputs
        return session.run(self.q_function, feed_dict={self.form: form, self.pos: pos, self.deprel: deprel})

    def target_policy(self, session, inputs):
        form, pos, deprel = inputs
        return session.run(self.tgt_q_function, feed_dict={self.form: form, self.pos: pos, self.deprel: deprel})

    def update_target(self, session):
        session.run(self.update_op)

    def classify(self, session, inputs):
        return self.policy(session, inputs)
