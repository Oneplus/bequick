#!/usr/bin/env python

def batch(iterable, n=1):
    """

    :param iterable:
    :param n:
    :return:
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]