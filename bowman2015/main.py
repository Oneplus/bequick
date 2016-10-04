#!/usr/bin/env python
import argparse


def train(train_data, devel_data, test_data, max_iteration=1):
    for iteration in range(max_iteration):
        pass


def load_data(path):
    """

    :param path: str, the path to the data file.
    :return: list(list(int))
    """
    ret = []
    for line in open(path, 'r'):
        tokens = line.strip().split('\t')
        ret.append(
            (int(tokens[0]), [int(i) for i in tokens[1].split()], [int(i) for i in tokens[2].split()])
        )
    return ret


def main():
    usage = "An implementation of A large annotated corpus for learning natural language inference"
    cmd = argparse.ArgumentParser(usage=usage)
    cmd.add_argument("train", help="the path to the training file.")
    cmd.add_argument("devel", help="the path to the development file.")
    cmd.add_argument("test", help="the path to the testing file.")
    args = cmd.parse_args()

    train_data = load_data(args.train)
    devel_data = load_data(args.devel)
    test_data = load_data(args.test)

if __name__ == "__main__":
    main()
