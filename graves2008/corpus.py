#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from bequick.utils import zip_open


def read_dataset(filename):
    """
    Read raw dataset from file
    :param filename: str, the path to the file.
    :return: list(dict)
    """
    dataset = []
    fpi = zip_open(filename)
    for raw_data in fpi.read().strip().split('\n'):
        data = []
        for token in raw_data.split():
            form, pos = token.rsplit('/', 1)
            data.append({'form': form, 'pos': pos})
        dataset.append(data)
    return dataset


def get_alphabet(dataset, keyword):
    """

    :param dataset:
    :param keyword:
    :return:
    """
    ret = {None: 0, 'UNK': 1}
    for data in dataset:
        for item in data:
            form = item[keyword]
            if form not in ret:
                ret[form] = len(ret)
    return ret
