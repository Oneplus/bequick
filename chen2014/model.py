#!/usr/bin/env python


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
        self.pos_dim = pos_dim,
        self.deprel_size = deprel_size
        self.deprel_dim = deprel_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def initialize_word_embeddings(self, embeddings):
        pass

    def train(self, X, Y):
        pass

    def classify(self, x):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass