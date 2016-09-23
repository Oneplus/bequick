#!/usr/bin/env python
from __future__ import print_function
import sys
import argparse
import logging
from bordes2013.alphabet import Alphabet

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


class Dataset(object):
    """
    The TransE dataset.
    - ent_alpha: the entity alphabet.
    - rel_alpha: the relation alphabet.
    - n_left_entities_of_: the counter of an entity e on the left side of a relation r.
    - n_right_entities_of_: the counter of an entity e on the right side of a relation r.
    - ratio_left_of:
    - ratio_right_of:

    Note: By default, the unknown entity and unknown relation is marked as 0.
    """
    def __init__(self):
        self.ent_alpha = Alphabet()
        self.rel_alpha = Alphabet()
        self.n_left_entities_of_ = {}
        self.n_right_entities_of_ = {}
        self.ratio_left_of_ = {}
        self.ratio_right_of_ = {}
        self.dataset = []

    def load(self, ent2id_path, rel2id_path, kb_path):
        """

        :param ent2id_path: str, the path of entity to id file.
        :param rel2id_path: str, the path of relation to id file.
        :param kb_path: str, the path to the kb file.
        """
        # Build the entity-id map
        for line in open(ent2id_path, 'r'):
            tokens = line.strip().split()
            assert len(tokens) == 2
            entity, i = tokens[0], int(tokens[1])
            self.ent_alpha.add(entity, i)

        # Build the relation-id map
        for line in open(rel2id_path, 'r'):
            tokens = line.strip().split()
            assert len(tokens) == 2
            relation, i = tokens[0], int(tokens[1])
            self.rel_alpha.add(relation, i)
            self.n_left_entities_of_[i] = {}
            self.n_right_entities_of_[i] = {}
            self.ratio_left_of_[i] = {}
            self.ratio_right_of_[i] = {}

        # Build the
        for line in open(kb_path, 'r'):
            tokens = line.strip().split()
            assert len(tokens) == 3
            head_entity, tail_entity, relation = tokens
            if head_entity not in self.ent_alpha:
                logging.warn("Head entity \"%s\" not in the entity alphabet." % head_entity)
            hid = self.ent_alpha.get(head_entity)

            if tail_entity not in self.ent_alpha:
                logging.warn("Tail entity \"%s\" not in the entity alphabet." % tail_entity)
            tid = self.ent_alpha.get(tail_entity)

            if relation not in self.rel_alpha:
                new_relation = ("-UNK-REL-%d" % self.rel_alpha.n)
                logging.warn("Relation \"%s\" not in the relation alphabet." % relation)
                logging.warn("Add \"%s\" into the relation alphabet." % new_relation)
                self.rel_alpha.add(new_relation)
                rid = self.rel_alpha.get(new_relation)
            else:
                rid = self.rel_alpha.get(relation)

            self.n_left_entities_of_[rid][hid] += 1
            self.n_right_entities_of_[rid][tid] += 1
            self.dataset.append((hid, tid, rid))

        for rid, n_entities in self.n_left_entities_of_.items():
            s_1 = len(n_entities)
            s_2 = sum(v for v in n_entities.values())
            self.ratio_left_of_[rid] = float(s_1) / s_2

        for rid, n_entities in self.n_right_entities_of_.items():
            s_1 = len(n_entities)
            s_2 = sum(v for v in n_entities.values())
            self.ratio_right_of_[rid] = float(s_1) / s_2


class Trainer(object):
    def __init__(self, lr, margin, method, ds):
        """

        :param lr: float, the learning rate.
        :param margin: float, the margin
        :param method: str, the training method.
        :param ds: Dataset, the dataset
        """
        self.lr = lr
        self.margin = margin
        self.method = method

    def train(self):
        pass


def learn():
    cmd = argparse.ArgumentParser()
    cmd.add_argument("-size", type=int, default=100, help="the size")
    cmd.add_argument("-margin", type=float, default=1., help="the margin")
    cmd.add_argument("-method", type=str, default="bern", help="the method: bern|unif")
    cmd.add_argument("-lr", type=float, default=0.001, help="the learning rate.")
    cmd.add_argument("-ent2id", default="data/ent2id.txt", help="the path to the entity mapping.")
    cmd.add_argument("-rel2id", default="data/rel2id.txt", help="the path to the relation mapping.")
    cmd.add_argument("-kb", default="data/kb.txt", help="the path to the knowledge-base.")
    args = cmd.parse_args()
    logging.info("size=%d" % args.size)
    logging.info("learning rate=%f" % args.lr)
    logging.info("margin=%f" % args.margin)
    logging.info("method=%s" % args.method)
    ds = Dataset()
    ds.load(args.ent2id, args.rel2id, args.kb)
    trainer = Trainer(args.lr, args.margin, args.method, ds)
    trainer.train()


def test():
    pass


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "learn":
        learn()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        print("%s [learn|test] options" % sys.argv[1], file=sys.stderr)

if __name__ == "__main__":
    main()
