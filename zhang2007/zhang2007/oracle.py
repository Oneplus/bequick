#!/usr/bin/env python
# -*- coding: utf-8 -*-


def get_gold_actions(words):
    """
    Get gold transition actions from a list of words

    Parameters
        words: The list of input words

    Returns
        ret: list of string 'j' or 's'
    """
    ret = []
    for word in words:
        chars = word.decode("utf-8")
        for idx, _ in enumerate(chars):
            if idx == len(chars) - 1:
                ret.append('s')
            else:
                ret.append('j')
    return ret

if __name__ == "__main__":
    assert list("jsjssjsjss") == get_gold_actions(['浦东', '开发', '与', '建设', '同步', '。'])
