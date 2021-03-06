#!/usr/bin/env python
import bz2
import gzip
import codecs


def batch(iterable, n=1):
    """

    :param iterable:
    :param n:
    :return:
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def zip_open(path):
    """

    :param path: str, the path to the file.
    :return:
    """
    if path.endswith('bz2'):
        fpi = bz2.BZ2File(path, 'r')
    elif path.endswith('gz'):
        fpi = gzip.open(path, 'r')
    else:
        fpi = codecs.open(path, 'r', encoding='utf-8')
    return fpi
