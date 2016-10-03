#!/usr/bin/env python


class Sentence(object):
    """
    The sentence. It can be either in the raw string form or indexed by alphabet.
    """
    def __init__(self, sentence, sentence_parse, sentence_binary_parse):
        self.sentence = sentence
        self.sentence_parse = sentence_parse
        self.sentence_binary_parse = sentence_binary_parse

