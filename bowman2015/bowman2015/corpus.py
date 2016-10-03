#!/usr/bin/env python
import json
from nltk.tree import Tree
from sentence import Sentence


class Instance(object):
    def __init__(self, line, is_json=True):
        """

        :param line: the context to the instance.
        """
        if is_json:
            self._load_json(line)
        else:
            # TSV
            self._load_tsv(line)

    def _load_json(self, line):
        payload = json.loads(line)
        self.gold_label = payload["gold_label"]
        self.caption_id = payload["captionID"]
        self.pair_id = payload["pairID"]
        self.annotator_labels = payload["annotator_labels"]
        self.raw_sentence1 = Sentence(
            sentence=payload["sentence1"],
            sentence_binary_parse=Tree.fromstring(payload["sentence1_binary_parse"]),
            sentence_parse=Tree.fromstring(payload["sentence1_parse"])
        )
        self.raw_sentence2 = Sentence(
            sentence=payload["sentence2"],
            sentence_binary_parse=Tree.fromstring(payload["sentence2_binary_parse"]),
            sentence_parse=Tree.fromstring(payload["sentence2_parse"])
        )

    def _load_tsv(self, line):
        tokens = line.strip().split('\t')
        assert len(tokens) == 14, "There should be 14 columns in the data file."
        self.gold_label = tokens[0]
        self.caption_id = tokens[7]
        self.pair_id = tokens[8]
        self.annotator_labels = tokens[9:]
        self.raw_sentence1 = Sentence(
            sentence=tokens[5],
            sentence_binary_parse=Tree.fromstring(tokens[1]),
            sentence_parse=Tree.fromstring(tokens[3])
        )
        self.raw_sentence2 = Sentence(
            sentence=tokens[6],
            sentence_binary_parse=Tree.fromstring(tokens[2]),
            sentence_parse=Tree.fromstring(tokens[4])
        )


class Corpus(object):
    def __init__(self):
        pass


    def load_training_data(self, path):
        """

        :param path:
        :return:
        """
        pass

    def load_test_data(self, path):
        """

        :param path:
        :return:
        """
        pass