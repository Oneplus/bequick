#!/usr/bin/env python
import math
import logging
import tensorflow as tf
from tb_parser import Parser

tf.set_random_seed(1234)


class Model(object):
    def __init__(self,
                 form_size,
                 form_dim,
                 pos_size,
                 pos_dim,
                 deprel_size,
                 deprel_dim,
                 hidden_dim,
                 output_dim,
                 lambda_):
        self.form_size = form_size
        self.form_dim = form_dim
        self.pos_size = pos_size
        self.pos_dim = pos_dim
        self.deprel_size = deprel_size
        self.deprel_dim = deprel_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # PLACEHOLDER: input and output
        self.form_inputs = tf.placeholder(tf.int32,
                shape=[None, len(Parser.FORM_NAMES),],
                name="form_i")
        self.pos_inputs = tf.placeholder(tf.int32,
                shape=[None, len(Parser.POS_NAMES),],
                name="pos_i")
        self.deprel_inputs = tf.placeholder(tf.int32,
                shape=[None, len(Parser.DEPREL_NAMES)],
                name="deprel_i")
        self.y = tf.placeholder(tf.float32,
                shape=[None, output_dim],
                name="y_o")

        # EMBEDDING in CPU
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            # embedding
            self.form_emb = tf.Variable(
                    self.random_uniform_matrix(self.form_size, self.form_dim),
                    name="form_emb")
            self.pos_emb = tf.Variable(
                    self.random_uniform_matrix(self.pos_size, self.pos_dim),
                    name="pos_emb")
            self.deprel_emb = tf.Variable(
                    self.random_uniform_matrix(self.deprel_size, self.deprel_dim),
                    name="deprel_emb")
            _input = tf.concat(1, [
                tf.reshape(
                    tf.nn.embedding_lookup(self.form_emb, self.form_inputs),
                    [-1, len(Parser.FORM_NAMES) * self.form_dim]
                    ),
                tf.reshape(
                    tf.nn.embedding_lookup(self.pos_emb, self.pos_inputs),
                    [-1, len(Parser.POS_NAMES) * self.pos_dim]
                    ),
                tf.reshape(
                    tf.nn.embedding_lookup(self.deprel_emb, self.deprel_inputs),
                    [-1, len(Parser.DEPREL_NAMES) * self.deprel_dim]
                    )
                ])

        # MLP 
        self.input_dim = (len(Parser.FORM_NAMES) * self.form_dim
                + len(Parser.POS_NAMES) * self.pos_dim
                + len(Parser.DEPREL_NAMES) * self.deprel_dim)

        self.W0 = tf.Variable(
                self.random_uniform_matrix(self.input_dim, self.hidden_dim),
                name="W0")
        self.b0 = tf.Variable(tf.zeros([self.hidden_dim]), name="b0")

        self.W1 = tf.Variable(
                self.random_uniform_matrix(self.hidden_dim, self.output_dim),
                name="W1")
        self.b1 = tf.Variable(tf.zeros([self.output_dim]), name="b1")

        # build network.
        _layer = tf.nn.relu(tf.add(tf.matmul(_input, self.W0), self.b0))
        self.pred = tf.nn.softmax(tf.add(tf.matmul(_layer, self.W1), self.b1))

        # REGULARIZER
        regularizer = lambda_ * (tf.nn.l2_loss(self.W0) +
                tf.nn.l2_loss(self.b0) +
                tf.nn.l2_loss(self.W1) +
                tf.nn.l2_loss(self.b1))
        
        # LOSS
        self.loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y)) + regularizer)
        self.optm = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(self.loss)

    def init(self):
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def initialize_word_embeddings(self, indices, matrix):
        _indices = [tf.to_int32(i) for i in indices]
        self.session.run(tf.scatter_update(self.form_emb, _indices, matrix))

    def train(self, X, Y):
        X_form = [_[0] for _ in X]
        X_pos = [_[1] for _ in X]
        X_deprel = [_[2] for _ in X]

        _, cost = self.session.run([self.optm, self.loss],
                feed_dict = {
                    self.form_inputs: X_form,
                    self.pos_inputs: X_pos,
                    self.deprel_inputs: X_deprel,
                    self.y: Y
                    })
        return cost

    def classify(self, X):
        X_form = [_[0] for _ in X]
        X_pos = [_[1] for _ in X]
        X_deprel = [_[2] for _ in X]

        prediction = self.session.run(self.pred,
                feed_dict = {
                    self.form_inputs: X_form,
                    self.pos_inputs: X_pos,
                    self.deprel_inputs: X_deprel,
                    })
        return prediction

    def save(self, path):
        pass

    def load(self, path):
        pass

    def random_uniform_matrix(self, n_rows, n_cols):
        width = math.sqrt(6. / (n_rows + n_cols))
        return tf.random_uniform((n_rows, n_cols), -width, width)