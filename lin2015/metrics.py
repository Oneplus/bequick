#!/usr/bin/env python
import numpy as np
from sklearn.metrics.cluster.supervised import check_clusterings
from sklearn.metrics import confusion_matrix


def many_to_one_score(labels_true, labels_pred):
    """

    :param labels_true: array-like, (n_samples,)
    :param labels_pred: array-like, (n_samples,)
    :return accuracy: float
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    matrix = confusion_matrix(labels_true, labels_pred)
    mapping = matrix.argmax(axis=0)
    return np.array(labels_true == mapping[labels_pred], dtype=np.float32).sum() / len(labels_pred)
