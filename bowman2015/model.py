#!/usr/bin/env python
import math
import tensorflow as tf
tf.set_random_seed(1234)


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
        self.X1 = tf.placeholder(tf.int32, (max_sentence1_steps, batch_size), name='X1')
        self.X2 = tf.placeholder(tf.int32, (max_sentence2_steps, batch_size), name='X2')
        self.Y = tf.placeholder(tf.int32, batch_size, name='Y')

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            width = math.sqrt(6. / (form_size + form_dim))
            self.form_emb = tf.Variable(
                tf.random_uniform([self.form_size, self.form_dim], -width, width),
                name="form_emb")

            def get_inputs(input_placeholder, max_steps):
                # inputs for the first sentence
                inputs = tf.nn.embedding_lookup(self.form_emb, input_placeholder)
                # shape(inputs) => (max_steps, batch_size, form_dim)
                inputs = tf.reshape(inputs, [batch_size, max_steps, self.form_dim])
                # shape(inputs) => (batch_size, max_steps, form_dim)
                inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, max_steps, inputs)]
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

        s1_fw_cell, s1_bw_cell = get_stacked_lstm_cell()
        s2_fw_cell, s2_bw_cell = get_stacked_lstm_cell()

        with tf.variable_scope('sentence1'):
            outputs1, _, _ = tf.nn.bidirectional_rnn(s1_fw_cell, s1_bw_cell, inputs1, dtype=tf.float32)
        with tf.variable_scope('sentence2'):
            outputs2, _, _ = tf.nn.bidirectional_rnn(s2_fw_cell, s2_bw_cell, inputs2, dtype=tf.float32)

        output = tf.concat(1, [outputs1[-1], outputs2[-1]])
        # shape(output) => (32, 400)
        output = tf.tanh(output)
        # shape(output) => (32, 400)

        # The MLP layer
        sentence_hidden_dim = self.hidden_dim * 4
        width = math.sqrt(6. / (sentence_hidden_dim * 2))
        self.W0 = tf.Variable(
            tf.random_uniform((sentence_hidden_dim, sentence_hidden_dim), -width, width),
            name="W0")
        # shape(W0) => (400, 400)
        self.b0 = tf.Variable(tf.zeros([sentence_hidden_dim]), name="b0")
        # shape(b0) => (400)
        width = math.sqrt(6. / (sentence_hidden_dim + self.output_dim))
        self.W1 = tf.Variable(
            tf.random_uniform((sentence_hidden_dim, self.output_dim), -width, width),
            name="W1")
        self.b1 = tf.Variable(tf.zeros([self.output_dim]), name="b1")
        logits = tf.matmul(tf.nn.relu(tf.matmul(output, self.W0) + self.b0), self.W1) + self.b1
        self.predication = tf.nn.softmax(logits)
        # shape(pred) => (32, 3)

        # LOSS
        reg = regularizer * (tf.reduce_sum(tf.nn.l2_loss(x) for x in [self.W0, self.b0, self.W1, self.b1]))
        self.loss = (tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.Y)) + reg)

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
        self.session.run(tf.initialize_all_variables())

    def initialize_word_embeddings(self, indices, matrix):
        _indices = [tf.to_int32(i) for i in indices]
        self.session.run(tf.scatter_update(self.form_emb, _indices, matrix))

    def train(self, X1, X2, Y):
        """

        :param X1:
        :param X2:
        :param Y:
        :return:
        """
        _, cost = self.session.run([self.optimization, self.loss], feed_dict={self.X1: X1, self.X2: X2, self.Y: Y})
        return cost

    def classify(self, X1, X2):
        """

        :param X1:
        :param X2:
        :return:
        """
        return self.session.run(self.predication, feed_dict={self.X1: X1, self.X2: X2})
