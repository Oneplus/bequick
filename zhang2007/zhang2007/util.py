#!/usr/bin/env python
from heapq import heappush, heappop, heappushpop


def chartype(ch):
    """ Get the type of character. """
    return 'other'


def number_of_characters(words):
    """
    Get number of characters in a list of words

    Parameters
    ----------
    words; list(str) the input word sequence

    Returns
    -------
    s: int
    """
    return sum(len(word.decode("utf-8")) for word in words)


def convert_words_to_characters(words):
    """
    Convert words into character sequence

    Parameters
    ----------
    words: list(str), the input word sequence

    Returns
    -------
    chars: list(str)
    """
    chars = []
    for word in words:
        chars.extend([ch.encode("utf-8") for ch in word.decode("utf-8")])
    return chars


def kmax_heappush(array, state, k):
    """
    Insert the state into a k-max array
    """
    if len(array) < k:
        heappush(array, state)
        return True
    elif len(array) == k:
        if array[0].score < state.score:
            heappushpop(array, state)
            return True
        return False
    return False
