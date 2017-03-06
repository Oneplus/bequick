#!/usr/bin/env python
import tensorflow as tf
import numpy as np
tf.set_random_seed(1234)


class Model(object):
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

    def _session(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.session.run(tf.global_variables_initializer())

    def initialize_word_embeddings(self, indices, matrix):
        self.session.run(tf.scatter_update(self.emb, indices, matrix))


class FlattenModel(Model):
    def __init__(self, algorithm, form_size, form_dim, hidden_dim, output_dim, max_steps, batch_size=1):
        self.algorithm = algorithm
        self.form_size = form_size
        self.form_dim = form_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.optimization, self.loss, self.prediction = None, None, None
        self.X, self.L, self.Y = None, None, None

    def _input_placeholder(self):
        with tf.name_scope('input'):
            X = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_steps), name="X")
            L = tf.placeholder(tf.int32, shape=(self.batch_size,), name="L")
            Y = tf.placeholder(tf.int32, shape=(self.batch_size,), name='Y')
        return X, L, Y

    def train(self, documents, lengths, labels):
        effective_n = documents.shape[0]
        if effective_n < self.batch_size:
            new_documents = np.zeros(shape=(self.batch_size, self.max_steps), dtype=np.int32)
            new_documents[: effective_n, ] = documents
            new_lengths = np.zeros(shape=self.batch_size, dtype=np.int32)
            new_lengths[: effective_n] = lengths
            new_labels = np.zeros(shape=self.batch_size, dtype=np.int32)
            new_labels[: effective_n] = labels
            documents, lengths, labels = new_documents, new_lengths, new_labels
        _, cost = self.session.run([self.optimization, self.loss],
                                   feed_dict={self.X: documents, self.L: lengths, self.Y: labels})
        return cost

    def classify(self, documents, lengths):
        effective_n = documents.shape[0]
        if effective_n < self.batch_size:
            new_documents = np.zeros(shape=(self.batch_size, self.max_steps), dtype=np.int32)
            new_documents[: effective_n, ] = documents
            new_lengths = np.zeros(shape=self.batch_size, dtype=np.int32)
            new_lengths[: effective_n] = lengths
            documents, lengths = new_documents, new_lengths
        ret = self.session.run(self.prediction, feed_dict={self.X: documents, self.L: lengths})
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
        self.document_expr = tf.reduce_sum(inputs, axis=1) / tf.expand_dims(tf.cast(self.L, tf.float32), 1)
        self.logits = self._mlp_op(self.document_expr)
        self.prediction = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimization = self._optimizer_op(self.loss)
        self._session()


class FlattenBiGRU(FlattenModel):
    def __init__(self, algorithm, n_layers, form_size, form_dim, hidden_dim, output_dim, max_steps, batch_size=1):
        FlattenModel.__init__(self, algorithm, form_size, form_dim, hidden_dim, output_dim, max_steps, batch_size)
        self.n_layers = n_layers
        self.X, self.L, self.Y = self._input_placeholder()

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.emb = tf.get_variable("emb", shape=(form_size, form_dim),
                                       initializer=tf.constant_initializer(0.), trainable=False)
        inputs = tf.nn.embedding_lookup(self.emb, self.X)
        # RNN for the 1st sentence.
        fw_cell = tf.contrib.rnn.GRUCell(hidden_dim)
        bw_cell = tf.contrib.rnn.GRUCell(hidden_dim)
        fw_stacked_cell = tf.contrib.rnn.MultiRNNCell([fw_cell] * n_layers, state_is_tuple=True)
        bw_stacked_cell = tf.contrib.rnn.MultiRNNCell([bw_cell] * n_layers, state_is_tuple=True)
        output_fw, output_bw = tf.nn.bidirectional_dynamic_rnn(fw_stacked_cell, bw_stacked_cell, inputs,
                                                               sequence_length=self.L, dtype=tf.float32)[0]
        indices_fw = tf.range(0, self.batch_size) * self.max_steps + (self.L - 1)
        indices_bw = tf.range(0, self.batch_size) * self.max_steps
        output_fw = tf.gather(tf.reshape(output_fw, [-1, self.hidden_dim]), indices_fw)
        output_bw = tf.gather(tf.reshape(output_bw, [-1, self.hidden_dim]), indices_bw)
        self.document_expr = tf.concat([output_fw, output_bw], 1)
        self.logits = self._mlp_op(self.document_expr)
        self.prediction = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimization = self._optimizer_op(self.loss)
        self._session()


class FlattenBiLSTM(FlattenModel):
    def __init__(self, algorithm, n_layers, form_size, form_dim, hidden_dim, output_dim, max_steps,
                 batch_size=1):
        FlattenModel.__init__(self, algorithm, form_size, form_dim, hidden_dim, output_dim, max_steps,
                              batch_size)
        self.n_layers = n_layers
        self.X, self.L, self.Y = self._input_placeholder()

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.emb = tf.get_variable("emb", shape=(form_size, form_dim),
                                       initializer=tf.constant_initializer(0.), trainable=False)
        inputs = tf.nn.embedding_lookup(self.emb, self.X)
        # inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(inputs, max_steps, 1)]
        # RNN for the 1st sentence.
        fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
        bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
        fw_stacked_cell = tf.contrib.rnn.MultiRNNCell([fw_cell] * n_layers, state_is_tuple=True)
        bw_stacked_cell = tf.contrib.rnn.MultiRNNCell([bw_cell] * n_layers, state_is_tuple=True)
        output_fw, output_bw = tf.nn.bidirectional_dynamic_rnn(fw_stacked_cell, bw_stacked_cell, inputs,
                                                               sequence_length=self.L, dtype=tf.float32)[0]
        indices_fw = tf.range(0, self.batch_size) * self.max_steps + (self.L - 1)
        indices_bw = tf.range(0, self.batch_size) * self.max_steps
        output_fw = tf.gather(tf.reshape(output_fw, [-1, self.hidden_dim]), indices_fw)
        output_bw = tf.gather(tf.reshape(output_bw, [-1, self.hidden_dim]), indices_bw)
        self.document_expr = tf.concat([output_fw, output_bw], 1)
        self.logits = self._mlp_op(self.document_expr)
        self.prediction = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimization = self._optimizer_op(self.loss)
        self._session()


class TreeModel(Model):
    def __init__(self, algorithm, form_size, form_dim, hidden_dim, output_dim, max_sentences, max_words,
                 batch_size):
        self.algorithm = algorithm
        self.form_size = form_size
        self.form_dim = form_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_sentences = max_sentences
        self.max_words = max_words
        self.batch_size = batch_size
        self.optimization, self.loss, self.prediction = None, None, None
        self.X, self.L, self.L2, self.Y = None, None, None, None

    def _input_placeholder(self):
        with tf.name_scope('input'):
            X = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_sentences, self.max_words), name="X")
            L = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_sentences), name="L")
            L2 = tf.placeholder(tf.int32, shape=self.batch_size, name='L2')
            Y = tf.placeholder(tf.int32, shape=self.batch_size, name='Y')
        return X, L, L2, Y

    def _avg_sentence(self, word_tensor):
        stacked_inputs = tf.reshape(word_tensor,
                                    [self.batch_size * self.max_sentences, self.max_words, self.form_dim])
        stacked_lengths = tf.reshape(self.L, [self.batch_size * self.max_sentences, 1])
        sentences = tf.reduce_sum(stacked_inputs, axis=1) / tf.cast(stacked_lengths, tf.float32)
        return tf.reshape(sentences, [self.batch_size, self.max_sentences, self.form_dim])

    def _bi_gru_sentence(self, word_tensor):
        n_rows = self.batch_size * self.max_sentences
        stacked_inputs = tf.reshape(word_tensor, [n_rows, self.max_words, self.form_dim])
        stacked_lengths = tf.reshape(self.L, [n_rows, ])
        with tf.name_scope('sentence'):
            fw_cell = tf.contrib.rnn.GRUCell(self.form_dim)
            bw_cell = tf.contrib.rnn.GRUCell(self.form_dim)
            fw_stacked_cell = tf.contrib.rnn.MultiRNNCell([fw_cell] * self.n_layers, state_is_tuple=True)
            bw_stacked_cell = tf.contrib.rnn.MultiRNNCell([bw_cell] * self.n_layers, state_is_tuple=True)
            output_fw, output_bw = tf.nn.bidirectional_dynamic_rnn(fw_stacked_cell, bw_stacked_cell,
                                                                   stacked_inputs,
                                                                   sequence_length=stacked_lengths,
                                                                   dtype=tf.float32,
                                                                   scope='sentence')[0]
        indices_fw = tf.range(0, n_rows) * self.max_words + (stacked_lengths - 1)
        indices_bw = tf.range(0, n_rows) * self.max_words
        output_fw = tf.gather(tf.reshape(output_fw, [-1, self.form_dim]), indices_fw)
        output_bw = tf.gather(tf.reshape(output_bw, [-1, self.form_dim]), indices_bw)
        return tf.reshape(tf.concat([output_fw, output_bw], 1), [self.batch_size, self.max_sentences, -1])

    def _avg_document(self, sentence_tensor):
        # Document:Averaged
        mask = tf.expand_dims(tf.to_float(tf.sequence_mask(self.L2, self.max_sentences)), 2)
        return tf.reduce_sum(sentence_tensor * mask, axis=1) / tf.cast(tf.expand_dims(self.L2, 1), tf.float32)

    def _bi_gru_document(self, sentence_tensor):
        with tf.name_scope('document'):
            fw_cell = tf.contrib.rnn.GRUCell(self.hidden_dim)
            bw_cell = tf.contrib.rnn.GRUCell(self.hidden_dim)
            fw_stacked_cell = tf.contrib.rnn.MultiRNNCell([fw_cell] * self.n_layers, state_is_tuple=True)
            bw_stacked_cell = tf.contrib.rnn.MultiRNNCell([bw_cell] * self.n_layers, state_is_tuple=True)
            output_fw, output_bw = tf.nn.bidirectional_dynamic_rnn(fw_stacked_cell, bw_stacked_cell,
                                                                   sentence_tensor,
                                                                   sequence_length=self.L2,
                                                                   dtype=tf.float32,
                                                                   scope='document')[0]
        indices_fw = tf.range(0, self.batch_size) * self.max_sentences + (self.L2 - 1)
        indices_bw = tf.range(0, self.batch_size) * self.max_sentences
        output_fw = tf.gather(tf.reshape(output_fw, [-1, self.hidden_dim]), indices_fw)
        output_bw = tf.gather(tf.reshape(output_bw, [-1, self.hidden_dim]), indices_bw)
        return tf.concat([output_fw, output_bw], 1)

    def train(self, documents, lengths, lengths2, labels):
        effective_n = documents.shape[0]
        if effective_n < self.batch_size:
            new_documents = np.zeros(shape=(self.batch_size, self.max_sentences, self.max_words), dtype=np.int32)
            new_documents[: effective_n, ] = documents
            new_lengths = np.ones(shape=(self.batch_size, self.max_sentences), dtype=np.int32)
            new_lengths[: effective_n, ] = lengths
            new_lengths2 = np.zeros(shape=self.batch_size, dtype=np.int32)
            new_lengths2[: effective_n] = lengths2
            new_labels = np.zeros(shape=self.batch_size, dtype=np.int32)
            new_labels[: effective_n] = labels
            documents, lengths, lengths2, labels = new_documents, new_lengths, new_lengths2, new_labels
        _, cost = self.session.run([self.optimization, self.loss], feed_dict={
            self.X: documents, self.L: lengths, self.L2: lengths2, self.Y: labels})
        return cost

    def classify(self, documents, lengths, lengths2):
        effective_n = documents.shape[0]
        if effective_n < self.batch_size:
            new_documents = np.zeros(shape=(self.batch_size, self.max_sentences, self.max_words), dtype=np.int32)
            new_documents[: effective_n, ] = documents
            new_lengths = np.ones(shape=(self.batch_size, self.max_sentences), dtype=np.int32)
            new_lengths[: effective_n, ] = lengths
            new_lengths2 = np.zeros(shape=self.batch_size, dtype=np.int32)
            new_lengths2[: effective_n] = lengths2
            documents, lengths, lengths2 = new_documents, new_lengths, new_lengths2
        ret = self.session.run(self.prediction, feed_dict={
            self.X: documents, self.L: lengths, self.L2: lengths2})
        if effective_n < self.batch_size:
            return ret.argmax(axis=1)[:effective_n]
        return ret.argmax(axis=1)


class TreeAveragePipeGRU(TreeModel):
    def __init__(self, algorithm, n_layers, form_size, form_dim, hidden_dim, output_dim, max_sentences, max_words,
                 batch_size):
        TreeModel.__init__(self, algorithm, form_size, form_dim, hidden_dim, output_dim, max_sentences, max_words,
                           batch_size)
        self.n_layers = n_layers
        self.X, self.L, self.L2, self.Y = self._input_placeholder()
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.emb = tf.get_variable("emb", shape=(form_size, form_dim),
                                       initializer=tf.constant_initializer(0.), trainable=False)
        inputs = tf.nn.embedding_lookup(self.emb, self.X)
        sentences = self._avg_sentence(inputs)
        self.document = self._bi_gru_document(sentences)
        self.logits = self._mlp_op(self.document)
        self.prediction = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimization = self._optimizer_op(self.loss)
        self._session()


class TreeGRUPipeAverage(TreeModel):
    def __init__(self, algorithm, n_layers, form_size, form_dim, hidden_dim, output_dim, max_sentences, max_words,
                 batch_size):
        TreeModel.__init__(self, algorithm, form_size, form_dim, hidden_dim, output_dim, max_sentences, max_words,
                           batch_size)
        self.n_layers = n_layers
        self.X, self.L, self.L2, self.Y = self._input_placeholder()
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.emb = tf.get_variable("emb", shape=(form_size, form_dim),
                                       initializer=tf.constant_initializer(0.), trainable=False)
        inputs = tf.nn.embedding_lookup(self.emb, self.X)
        sentences = self._bi_gru_sentence(inputs)
        self.document = self._avg_document(sentences)
        self.logits = self._mlp_op(self.document)
        self.prediction = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimization = self._optimizer_op(self.loss)
        self._session()


class TreeGRUPipeGRU(TreeModel):
    def __init__(self, algorithm, n_layers, form_size, form_dim, hidden_dim, output_dim, max_sentences, max_words,
                 batch_size):
        TreeModel.__init__(self, algorithm, form_size, form_dim, hidden_dim, output_dim, max_sentences, max_words,
                           batch_size)
        self.n_layers = n_layers
        self.X, self.L, self.L2, self.Y = self._input_placeholder()
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.emb = tf.get_variable("emb", shape=(form_size, form_dim),
                                       initializer=tf.constant_initializer(0.), trainable=False)
        inputs = tf.nn.embedding_lookup(self.emb, self.X)
        sentences = self._bi_gru_sentence(inputs)
        self.document = self._bi_gru_document(sentences)
        self.logits = self._mlp_op(self.document)
        self.prediction = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimization = self._optimizer_op(self.loss)
        self._session()
