#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import gzip


def _selectional_open(filename):
    if filename.lower() in ("changelog", "readme", "readme.md"):
        return None
    if filename.endswith("gz"):
        try:
            fp = gzip.open(filename, "r")
        except IOError:
            print("ERROR: failed to open file.", file=sys.stderr)
            return None
    else:
        try:
            fp = open(filename, "r")
        except IOError:
            print("ERROR: failed to open file.", file=sys.stderr)
            return None
    return fp


def read_targets(filename):
    """
    Read the target file. Each token on line and the none target
    is annotated as `_`.

    Parameters
    ----------
    filename: str
        The path to the filename.

    Returns
    -------
    the dataset.
    """
    dataset = open(filename, "r").read().strip().split("\n\n")
    return [data.split("\n") for data in dataset]


def read_syntax(filename):
    """

    :param filename:
    :return:
    """
    fp = _selectional_open(filename)
    if fp is None:
        return
    dataset = fp.read().strip().split("\n\n")
    return [data.split("\n") for data in dataset]


def read_props(filename):
    fp = _selectional_open(filename)
    if fp is None:
        return
    dataset = fp.read().strip().split("\n\n")
    return [data.split("\n") for data in dataset]


def read_words(filename):
    fp = _selectional_open(filename)
    if fp is None:
        return
    dataset = fp.read().strip().split("\n\n")
    return [data.split("\n") for data in dataset]


def read_dir(dirname, function):
    output = None
    for (cur, dirs, files) in os.walk(dirname):
        output = []
        for f in sorted(files):
            output.extend(function(os.path.join(cur, f)))
        break
    return output


def read_props_dir(dirname):
    return read_dir(dirname, read_props)


def read_words_dir(dirname):
    return read_dir(dirname, read_words)


def read_syntax_dir(dirname):
    return read_dir(dirname, read_syntax)


def read_targets_dir(dirname):
    return read_dir(dirname, read_targets)
