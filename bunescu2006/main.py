#!/usr/bin/env python
import sys
import argparse
import logging


class MentionDictionary(object):
    NME = '--NME--'
    LOG = logging.getLogger(__file__)

    def __init__(self, path):
        """

        :param path:
        """
        self.mentions = {}
        for line in open(path, 'r'):
            tokens = line.strip().split('\t')
            self.mentions[tokens[0]] = tokens[1]
        self.LOG.info("Loaded %d (mention, entity) pair" % len(self.mentions))

    def get(self, name):
        """

        :param name: str, The name of the mention.
        :return: str, The name of the entity.
        """
        return self.mentions.get(name, self.NME)


class Vocabulary(object):
    UNK = '__UNK__'
    LOG = logging.getLogger(__file__)

    def __init__(self, path):
        """

        :param path:
        """
        self.vocab = {self.UNK: 0}
        idx = 1
        for word in open(path, 'r'):
            word = word.strip()
            self.vocab[word] = idx
            idx += 1
        self.LOG.info("Loaded %d words to vocabulary." % len(self.vocab))

    def get(self, words):
        """

        :param words: list(str),
        :return: list(int),
        """
        return [self.vocab.get(word, 0) for word in words]


if __name__ == "__main__":
    cmd = argparse.ArgumentParser('An implementation of Using Encyclopedic Knowledge for Named Entity Disambiguation')
    cmd.add_argument('mentions', help='The path to the (mention, entity) dictionary.')
    cmd.add_argument('vocabulary', help='The vocabulary.')
    cmd.add_argument('title', help='The path to the title file.')
    args = cmd.parse_args()

    mentions = MentionDictionary(args.mentions)
    vocabulary = Vocabulary(args.vocabulary)
