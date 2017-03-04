#!/usr/bin/env python
import tensorflow as tf
import numpy as np
tf.set_random_seed(1234)


class FlattenModel(object):
    def __init__(self, algorithm, form_size, form_dim, hidden_dim, output_dim, max_steps, batch_size=1):
        self.algorithm = algorithm
        self.form_size = form_size
        self.form_dim = form_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_steps = max_steps
        self.batch_size = batch_size

    def _input_placeholder(self):
        with tf.name_scope('input'):
            X = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_steps), name="X")
            L = tf.placeholder(tf.float32, shape=(self.batch_size,), name="L")
            Y = tf.placeholder(tf.int32, shape=(self.batch_size,), name='Y')
        return X, L, Y

    def _optimizer_op(self, loss):
        if self.algorithm == "adagrad":
            optimization = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss)
        elif self.algorithm == "adadelta":
            optimization = tf.train.AdadeltaOptimizer(learning_rate=0.01).minimize(loss)
        elif self.algorithm == "adam":
            optimization = tf.train.AdamOptimizer().minimize(loss)
        else:
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(0.01, global_step, 100000, 0.96)
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            variables = tf.trainable_variables()
            gradients, _ = tf.clip_by_global_norm(tf.gradients(loss, variables), 5.)
            optimization = opt.apply_gradients(zip(gradients, variables), global_step=global_step)
        return optimization

    def _mlp_op(self, document_expr):
        hidden_layer = tf.contrib.layers.fully_connected(document_expr, self.hidden_dim, activation_fn=tf.nn.relu,
                                                         weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                         biases_initializer=tf.constant_initializer(0.))
        logits = tf.contrib.layers.fully_connected(hidden_layer, self.output_dim, activation_fn=None,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                   biases_initializer=tf.constant_initializer(0.))
        return logits

    def initialize_word_embeddings(self, indices, matrix):
        self.session.run(tf.scatter_update(self.emb, indices, matrix))

    def train(self, X, L, Y):
        effective_n = X.shape[0]
        if effective_n < self.batch_size:
            new_X = np.zeros(shape=(self.batch_size, self.max_steps), dtype=np.int32)
            new_X[: effective_n, ] = X
            new_L = np.zeros(shape=self.batch_size, dtype=np.int32)
            new_L[: effective_n] = L
            new_Y = np.zeros(shape=self.batch_size, dtype=np.int32)
            new_Y[: effective_n] = Y
            X, L, Y = new_X, new_L, new_Y
        _, cost = self.session.run([self.optimization, self.loss], feed_dict={self.X: X, self.L: L, self.Y: Y})
        return cost

    def classify(self, X, L):
        effective_n = X.shape[0]
        if effective_n < self.batch_size:
            new_X = np.zeros(shape=(self.batch_size, self.max_steps), dtype=np.int32)
            new_X[: effective_n,] = X
            new_L = np.zeros(shape=self.batch_size, dtype=np.int32)
            new_L[: effective_n] = L
            X, L = new_X, new_L
        ret = self.session.run(self.prediction, feed_dict={self.X: X, self.L: L})
        if effective_n < self.batch_size:
            return ret.argmax(axis=1)[:effective_n]
        return ret.argmax(axis=1)


class FlattenAverage(FlattenModel):
    def __init__(self, algorithm, form_size, form_dim, hidden_dim, output_dim, max_steps, batch_size=1):
        FlattenModel.__init__(self, algorithm, form_size, form_dim, hidden_dim, output_dim, max_steps, batch_size)
        self.X, self.L, self.Y = self._input_placeholder()

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.emb = tf.get_variable("emb", shape=(form_size, form_dim),
                                       initializer=tf.constant_initializer(0.), trainable=False)
            inputs = tf.nn.embedding_lookup(self.emb, self.X)
            inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(inputs, max_steps, 1)]

        self.document_expr = tf.reduce_sum(inputs, axis=0) / tf.expand_dims(self.L, 1)
        self.logits = self._mlp_op(self.document_expr)
        self.prediction = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimization = self._optimizer_op(self.loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())


class FlattenBiGRU(FlattenModel):
    def __init__(self, algorithm, n_layers, form_size, form_dim, hidden_dim, output_dim, max_steps, batch_size=1):
        FlattenModel.__init__(self, algorithm, form_size, form_dim, hidden_dim, output_dim, max_steps, batch_size)
        self.n_layers = n_layers
        self.X, self.L, self.Y = self._input_placeholder()

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.emb = tf.get_variable("emb", shape=(form_size, form_dim),
                                       initializer=tf.constant_initializer(0.), trainable=False)
            inputs = tf.nn.embedding_lookup(self.emb, self.X)
            inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(inputs, max_steps, 1)]

        # RNN for the 1st sentence.
        fw_lstm_cell = tf.contrib.rnn.GRUCell(hidden_dim)
        bw_lstm_cell = tf.contrib.rnn.GRUCell(hidden_dim)
        fw_stacked_lstm_cell = tf.contrib.rnn.MultiRNNCell([fw_lstm_cell] * n_layers, state_is_tuple=True)
        bw_stacked_lstm_cell = tf.contrib.rnn.MultiRNNCell([bw_lstm_cell] * n_layers, state_is_tuple=True)
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_stacked_lstm_cell, bw_stacked_lstm_cell, inputs,
                                                                sequence_length=self.L, dtype=tf.float32)
        output_bw = tf.split(outputs[0], 2, 1)[1]
        output_fw = tf.split(outputs[-1], 2, 1)[0]
        self.document_expr = tf.concat([output_fw, output_bw], 1)
        self.logits = self._mlp_op(self.document_expr)
        self.prediction = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimization = self._optimizer_op(self.loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())


class FlattenBiLSTM(FlattenModel):
    def __init__(self, algorithm, n_layers, form_size, form_dim, hidden_dim, output_dim, max_steps, batch_size=1):
        FlattenModel.__init__(self, algorithm, form_size, form_dim, hidden_dim, output_dim, max_steps, batch_size)
        self.n_layers = n_layers
        self.X, self.L, self.Y = self._input_placeholder()

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.emb = tf.get_variable("emb", shape=(form_size, form_dim),
                                       initializer=tf.constant_initializer(0.), trainable=False)
            inputs = tf.nn.embedding_lookup(self.emb, self.X)
            inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(inputs, max_steps, 1)]

        # RNN for the 1st sentence.
        fw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
        bw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
        fw_stacked_lstm_cell = tf.contrib.rnn.MultiRNNCell([fw_lstm_cell] * n_layers, state_is_tuple=True)
        bw_stacked_lstm_cell = tf.contrib.rnn.MultiRNNCell([bw_lstm_cell] * n_layers, state_is_tuple=True)
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_stacked_lstm_cell, bw_stacked_lstm_cell, inputs,
                                                                sequence_length=self.L, dtype=tf.float32)
        output_bw = tf.split(outputs[0], 2, 1)[1]
        output_fw = tf.split(outputs[-1], 2, 1)[0]
        self.document_expr = tf.concat([output_fw, output_bw], 1)
        self.logits = self._mlp_op(self.document_expr)
        self.prediction = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimization = self._optimizer_op(self.loss)


class DocumentTreeModel(object):
    def __init__(self, algorithm, n_layers, form_size, form_dim, hidden_dim, output_dim, max_document, max_sentence,
                 batch_size=1):
        self.algorithm = algorithm
        self.n_layers = n_layers
        self.form_size = form_size
        self.form_dim = form_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_document = max_document
        self.max_sentence = max_sentence
        self.batch_size = batch_size

    def _input_placeholder(self):
        with tf.name_scope('input'):
            X = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_document, self.max_sentence), name="X")
            L = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_document), name="L")
            Y = tf.placeholder(tf.int32, shape=(self.batch_size,), name='Y')
        return X, L, Y


class DocumentTreeAverage(DocumentTreeModel):
    def __init__(self, algorithm, n_layers, form_size, form_dim, hidden_dim, output_dim, max_document, max_sentence,
                 batch_size=1):
        DocumentTreeModel.__init__(algorithm, form_size, form_dim, hidden_dim, output_dim, max_document, max_sentence,
                                   batch_size)
