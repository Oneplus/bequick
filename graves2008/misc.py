#!/usr/bin/env python
import numpy as np
from itertools import chain


def transform(dataset, form_alphabet, pos_alphabet, max_steps):
    """
    transform a dataset into numeric form.

    :param dataset: list, a list of data
    :param form_alphabet: dict, the form alphabet.
    :param pos_alphabet: dict, the postag alphabet.
    :param max_steps: int, the number of maximum steps
    :return: np,array, np.array
    """
    n_instances = len(dataset)
    x = np.zeros((n_instances, max_steps), dtype=np.int32)
    y = np.zeros((n_instances, max_steps), dtype=np.int32)
    steps = np.zeros(n_instances, dtype=np.int32)
    for i, data in enumerate(dataset):
        x_ = np.array([form_alphabet.get(token['form'], 1) for token in data], dtype=np.int32)
        y_ = np.array([pos_alphabet.get(token['pos']) for token in data], dtype=np.int32)
        x[i, : len(x_)] = x_
        y[i, : len(y_)] = y_
        steps[i] = len(x_)
    return x, y, steps


def get_max_steps(train_data, devel_data, test_data):
    max_steps = 0
    for data in chain(train_data, devel_data, test_data):
        max_steps = max(max_steps, len(data))
    return max_steps
