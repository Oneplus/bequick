#!/usr/bin/env python
import math
import logging
import tensorflow as tf
from tb_parser import Parser

class Model(object):
    def __init__(self,
                 form_size,
                 form_dim,
                 pos_size,
                 pos_dim,
                 deprel_size,
                 deprel_dim,
                 hidden_dim,
                 output_dim):
        self.form_size = form_size
        self.form_dim = form_dim
        self.pos_size = pos_size
        self.pos_dim = pos_dim
        self.deprel_size = deprel_size
        self.deprel_dim = deprel_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # embedding
        width = 6. / math.sqrt(form_size + form_dim)
        self.form_emb = tf.Variable(
                tf.random_uniform([self.form_size, self.form_dim], -width, width),
                name="form_emb")

        width = 6. / math.sqrt(pos_size + pos_dim)
        self.pos_emb = tf.Variable(
                tf.random_uniform([self.pos_size, self.pos_dim], -width, width),
                name="pos_emb")

        width = 6. / math.sqrt(deprel_size + deprel_dim)
        self.deprel_emb = tf.Variable(
                tf.random_uniform([self.deprel_size, self.deprel_dim], -width, width),
                name="deprel_emb")

        # MLP weight
        self.input_dim = (len(Parser.FORM_NAMES) * self.form_dim
                + len(Parser.POS_NAMES) * self.pos_dim
                + len(Parser.DEPREL_NAMES) * self.deprel_dim)
        width = 6. / math.sqrt(self.input_dim + hidden_dim)
        self.W0 = tf.Variable(
                tf.random_uniform([self.input_dim, self.hidden_dim], -width, width),
                name="W0")
        self.b0 = tf.Variable(tf.zeros([self.hidden_dim]), name="b0")

        width = 6. / math.sqrt(hidden_dim + output_dim)
        self.W1 = tf.Variable(
                tf.random_uniform([self.hidden_dim, self.output_dim], -width, width),
                name="W1")
        self.b1 = tf.Variable(tf.zeros([self.output_dim]), name="b1")

        # input and output
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

        # build network.
        _input = tf.reshape(
                tf.concat(1, [
                    tf.nn.embedding_lookup(self.form_emb, self.form_inputs),
                    tf.nn.embedding_lookup(self.pos_emb, self.pos_inputs),
                    tf.nn.embedding_lookup(self.deprel_emb, self.deprel_inputs)]
                    ),
                [-1, self.input_dim]
                )
        _layer = tf.nn.relu(tf.add(tf.matmul(_input, self.W0), self.b0))
        self.pred = tf.nn.softmax(tf.add(tf.matmul(_layer, self.W1), self.b1))

        # loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y))
        self.optm = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)

    def initialize_word_embeddings(self, embeddings):
        pass

    def init(self):
        init = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(init)

    def train(self, X, Y):
        X_form = [_[0] for _ in X]
        X_pos = [_[1] for _ in X]
        X_deprel = [_[2] for _ in X]

        self.session.run(self.optm,
                feed_dict = {
                    self.form_inputs: X_form,
                    self.pos_inputs: X_pos,
                    self.deprel_inputs: X_deprel,
                    self.y: Y
                    })

        cost = self.session.run(self.loss,
                feed_dict = {
                    self.form_inputs: X_form,
                    self.pos_inputs: X_pos,
                    self.deprel_inputs: X_deprel,
                    self.y: Y
                    })
        return cost

    def classify(self, x):
        x_form, x_pos, x_deprel = x
        prediction = self.session.run(self.pred,
                feed_dict = {
                    self.form_inputs: [x_form],
                    self.pos_inputs: [x_pos],
                    self.deprel_inputs: [x_deprel],
                    })
        return prediction

    def save(self, path):
        pass

    def load(self, path):
        pass
