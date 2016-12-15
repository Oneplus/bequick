#!/usr/bin/env python
import tensorflow as tf
import math


def random_uniform_matrix(n_rows, n_cols):
    width = math.sqrt(6. / (n_rows + n_cols))
    return tf.random_uniform((n_rows, n_cols), -width, width)
