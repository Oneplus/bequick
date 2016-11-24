#!/usr/bin/env python
import math
import logging
import tensorflow as tf
from tb_parser import Parser

tf.set_random_seed(1234)


def random_uniform_matrix(n_rows, n_cols):
    width = math.sqrt(6. / (n_rows + n_cols))
    return tf.random_uniform((n_rows, n_cols), -width, width)


class Network(object):
    def __init__(self, form_size, form_dim, pos_size, pos_dim, deprel_size, deprel_dim, hidden_dim, output_dim):
        self.form_size = form_size
        self.form_dim = form_dim
        self.pos_size = pos_size
        self.pos_dim = pos_dim
        self.deprel_size = deprel_size
        self.deprel_dim = deprel_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # input and output
        self.form_inputs = tf.placeholder(tf.int32, shape=(None, len(Parser.FORM_NAMES), ), name="form_i")
        self.pos_inputs = tf.placeholder(tf.int32, shape=(None, len(Parser.POS_NAMES), ), name="pos_i")
        self.deprel_inputs = tf.placeholder(tf.int32, shape=(None, len(Parser.DEPREL_NAMES), ), name="deprel_i")
        self.y = tf.placeholder(tf.int32, shape=(None, ), name="y_o")

        form_features_dim = len(Parser.FORM_NAMES) * self.form_dim
        pos_features_dim = len(Parser.POS_NAMES) * self.pos_dim
        deprel_features_dim = len(Parser.DEPREL_NAMES) * self.deprel_dim
        # EMBEDDING in CPU
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            # embedding
            self.form_emb = tf.Variable(random_uniform_matrix(self.form_size, self.form_dim), name="form_emb")
            self.pos_emb = tf.Variable(random_uniform_matrix(self.pos_size, self.pos_dim), name="pos_emb")
            self.deprel_emb = tf.Variable(random_uniform_matrix(self.deprel_size, self.deprel_dim), name="deprel_emb")
            _input = tf.concat(1, [
                tf.reshape(tf.nn.embedding_lookup(self.form_emb, self.form_inputs), [-1, form_features_dim]),
                tf.reshape(tf.nn.embedding_lookup(self.pos_emb, self.pos_inputs), [-1, pos_features_dim]),
                tf.reshape(tf.nn.embedding_lookup(self.deprel_emb, self.deprel_inputs), [-1, deprel_features_dim])
                ])

        # MLP
        self.input_dim = form_features_dim + pos_features_dim + deprel_features_dim
        self.W0 = tf.Variable(random_uniform_matrix(self.input_dim, self.hidden_dim), name="W0")
        self.b0 = tf.Variable(tf.zeros([self.hidden_dim]), name="b0")
        self.W1 = tf.Variable(random_uniform_matrix(self.hidden_dim, self.output_dim), name="W1")
        self.b1 = tf.Variable(tf.zeros([self.output_dim]), name="b1")
        _layer = tf.nn.relu(tf.add(tf.matmul(_input, self.W0), self.b0))
        self.prediction = tf.add(tf.matmul(_layer, self.W1), self.b1)

    def initialize_word_embeddings(self, session, indices, matrix):
        _indices = [tf.to_int32(i) for i in indices]
        session.run(tf.scatter_update(self.form_emb, _indices, matrix))

    def l2_loss(self):
        return (tf.nn.l2_loss(self.W0) + tf.nn.l2_loss(self.b0) +
                tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.b1))

    def get_training_feed(self, form, pos, deprel, Y):
        return {self.form_inputs: form, self.pos_inputs: pos, self.deprel_inputs: deprel, self.y: Y}

    def get_testing_feed(self, form, pos, deprel):
        return {self.form_inputs: form, self.pos_inputs: pos, self.deprel_inputs: deprel}


class SupervisedModel(object):
    def __init__(self, form_size, form_dim, pos_size, pos_dim, deprel_size, deprel_dim, hidden_dim, output_dim, l):
        self.network = Network(form_size, form_dim, pos_size, pos_dim, deprel_size, deprel_dim, hidden_dim, output_dim)

        regularizer = l * self.network.l2_loss()
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(self.network.prediction, self.network.y)) + regularizer
        self.optimization = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(self.loss)

    def train(self, session, X, Y):
        form = [_[0] for _ in X]
        pos = [_[1] for _ in X]
        deprel = [_[2] for _ in X]

        _, cost = session.run([self.optimization, self.loss],
                              feed_dict=self.network.get_training_feed(form, pos, deprel, Y))
        return cost

    def classify(self, session, X):
        form = [_[0] for _ in X]
        pos = [_[1] for _ in X]
        deprel = [_[2] for _ in X]

        prediction = session.run(self.network.prediction, feed_dict=self.network.get_testing_feed(form, pos, deprel))
        return prediction


class QlearningModel(object):
    def __init__(self, form_size, form_dim, pos_size, pos_dim, deprel_size, deprel_dim, hidden_dim, output_dim, l):
        self.network = Network(form_size, form_dim, pos_size, pos_dim, deprel_size, deprel_dim, hidden_dim, output_dim)
        self.pick = tf.placeholder(tf.int32, shape=(None,), name="y_o")
        regularizer = l * self.network.l2_loss()
        self.loss = tf.reduce_mean(
            tf.square(tf.sub(tf.gather(self.network.prediction, self.pick), self.network.y))) + regularizer
        self.optimization = tf.train.RMSPropOptimizer(learning_rate=0.00025, momentum=0.95).minimize(self.loss)

    def train(self, session, X, pick, Y):
        form = [_[0] for _ in X]
        pos = [_[1] for _ in X]
        deprel = [_[2] for _ in X]

        feed = self.network.get_training_feed(form, pos, deprel, Y)
        feed[self.pick] = pick
        _, cost = session.run([self.optimization, self.loss], feed_dict=feed)
        return cost

    def classify(self, session, X):
        form = [_[0] for _ in X]
        pos = [_[1] for _ in X]
        deprel = [_[2] for _ in X]

        prediction = session.run(self.network.prediction, feed_dict=self.network.get_testing_feed(form, pos, deprel))
        return prediction
