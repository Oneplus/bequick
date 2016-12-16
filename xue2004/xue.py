#!/usr/bin/env python
from __future__ import print_function
import argparse
import sys
import pickle
import logging
from xue2004.io_utils import read_syntax_dir, read_props_dir, read_words_dir
from xue2004.props import get_predicate_positions, get_arguments_of_predicate, locate_predicate
from xue2004.targets import get_target_indices
from xue2004.head_finder import CollinsHeadFinder
from xue2004.feature import extract_feature
from xue2004.constituent import build_constituent_tree, generate_candidate_constituents
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
LOG = logging.getLogger("xue2004")
DETAILS = open("log", "w")


def init_learn_options():
    parser = argparse.ArgumentParser("the learning component of xue2004.")
    parser.add_argument("-s", "--synt", dest="synt", help="use to specify the syntax file.")
    parser.add_argument("-p", "--props", dest="props", help="use to specify the prop file.")
    parser.add_argument("-w", "--words", dest="words", help="use to specify the words file.")
    parser.add_argument("-m", "--model", dest="model", help="use to specify the model path.")
    parser.add_argument("--simplified", action="store_true", default=False, help="perform C-* simplification.")
    return parser


def init_labeling_options():
    parser = argparse.ArgumentParser("the labeling component of xue2004.")
    parser.add_argument("-s", "--syntax", dest="synt", help="use to specify the syntax file.")
    parser.add_argument("-t", "--targets", dest="targets", help="use to specify the target file.")
    parser.add_argument("-w", "--words", dest="words", help="use to specify the input words.")
    parser.add_argument("-m", "--model", dest="model", help="use to specify the model path.")
    return parser


def validation():
    parser = init_learn_options()
    opts, args = parser.parse_args()

    # Reading the data.
    LOG.info("trace: reading data. ")
    words = read_words_dir(opts.words)
    syntax = read_syntax_dir(opts.synt)
    props = read_props_dir(opts.props)
    LOG.info("[done]")

    if len(syntax) != len(props) or len(syntax) != len(words):
        LOG.info("error: #syntax not equals to #props")
        return

    LOG.info("trace: read in %d sentence(s)." % len(words))
    hf = CollinsHeadFinder()
    LOG.info("trace: checking instance: 0")

    nr_predicates, nr_multi_verb_predicate = 0, 0
    nr_miss_located_predicate = 0
    for idx, (word, synt, prop) in enumerate(zip(words, syntax, props)):
        current_proportion = int(float(idx)/len(words) * 10)
        previous_proportion = int(float(idx - 1)/len(words) * 10)
        if current_proportion != previous_proportion:
            LOG.info(current_proportion)

        assert len(word) == len(synt)
        tree = build_constituent_tree(word, synt)[0]
        print(idx, tree.pprint(), file=DETAILS)
        hf.run(tree)
        positions = get_predicate_positions(prop)
        nr_predicates += len(positions)
        for predicate_position in positions:
            if predicate_position[0] + 1 != predicate_position[1]:
                print("%d multi-verb predicates - %d,%d" % (idx, predicate_position[0], predicate_position[1]),
                      file=DETAILS)
                nr_multi_verb_predicate += 1

            predicate = locate_predicate(tree, predicate_position)
            if predicate is None:
                print("%d missed predicate - %d,%d" % (idx, predicate_position[0], predicate_position[1]),
                      file=DETAILS)
                nr_miss_located_predicate += 1

    LOG.info("[done]")
    LOG.info("trace: # predicates =", nr_predicates)
    LOG.info("trace: # multi-verb predicates =", nr_multi_verb_predicate)
    LOG.info("trace: # miss located predicates =", nr_miss_located_predicate)


def learn():
    parser = init_learn_options()
    opts, args = parser.parse_args()

    # Reading the data.
    LOG.info("trace: reading data. ")
    words = read_words_dir(opts.words)
    syntax = read_syntax_dir(opts.synt)
    props = read_props_dir(opts.props)
    LOG.info("[done]")

    if len(syntax) != len(props) or len(syntax) != len(words):
        LOG.info("error: #syntax not equals to #props")
        return

    LOG.info("trace: read in %d sentence(s)." % len(words))
    hf = CollinsHeadFinder()
    LOG.info("trace: extracting features: 0")
    X, Y = [], []
    for idx, (word, synt, prop) in enumerate(zip(words, syntax, props)):
        current_proportion = int(float(idx)/len(words) * 10)
        previous_proportion = int(float(idx - 1)/len(words) * 10)
        if current_proportion != previous_proportion:
            LOG.info(current_proportion)

        tree = build_constituent_tree(word, synt)[0]
        hf.run(tree)
        positions = get_predicate_positions(prop)

        for predicate_position in positions:
            predicate = locate_predicate(tree, predicate_position)
            if predicate is None:
                LOG.info("warn: cannot locate predicate, #%d" % idx)
                continue

            args = get_arguments_of_predicate(prop, predicate)
            cons = generate_candidate_constituents(tree, predicate)

            for con in cons:
                found = False
                for key, start, end in args:
                    if start == con.start and end == con.end:
                        found = True
                        break
                X.append(extract_feature(predicate, con))
                Y.append((1, key) if found else (0, None))
    attrs = DictVectorizer()
    X1 = attrs.fit_transform([{xx: 1 for xx in x} for x in X])
    LOG.info("[done]")
    LOG.info("trace: extract %d feature(s)" % len(attrs.vocabulary_))
    LOG.info("trace: generate %d training instance(s)." % len(X))
    LOG.info("trace: number of positive instance(s) is %d" % sum([y[0] for y in Y]))
    LOG.info("trace: learning argument identification model.")
    model_phase_one = LogisticRegression(C=1)
    model_phase_one.fit(X1, [y[0] for y in Y])
    LOG.info("[done]")

    X2 = attrs.transform([{x: 1 for x in X[i]} for i in range(len(X)) if Y[i][0] == 1])
    labels = {}
    if opts.simplified:
        Y = [(y, (lab[2:] if y == 1 and lab.startswith("C-AM") else lab)) for y, lab in Y]
    for y, lab in Y:
        if y == 1 and lab not in labels:
            labels[lab] = len(labels)

    Y2 = [labels[y[1]] for y in Y if y[0] == 1]
    LOG.info("trace: classifying %d types" % len(labels))
    LOG.info("trace: learning argument classification model.")
    model_phase_two = LogisticRegression(C=1)
    model_phase_two.fit(X2, Y2)
    LOG.info("[done]")

    # pickle.dump((labels, attrs, model_phase_one, model_phase_two),
    #        gzip.open(opts.model, "w"))
    pickle.dump((labels, attrs, model_phase_one, model_phase_two), open(opts.model, "wb"))


def labeling():
    parser = init_labeling_options()
    opts, args = parser.parse_args()

    LOG.info("trace: reading data.")
    words = read_words_dir(opts.words)
    syntax = read_syntax_dir(opts.synt)
    targets = read_props_dir(opts.targets)
    LOG.info("[done]")

    if len(syntax) != len(targets) or len(syntax) != len(words):
        LOG.info("error: #syntax not equals to #props")
        return

    LOG.info("trace: loading model.")
    labels, attrs, model_phase_one, model_phase_two = pickle.load(open(opts.model, "rb"))
    LOG.info("[done]")

    inversed_labels = {}
    for label in labels:
        inversed_labels[labels[label]] = label
    hf = CollinsHeadFinder()
    LOG.info("trace: predicting: 0")
    for idx, (word, synt, target) in enumerate(zip(words, syntax, targets)):
        current_proportion = int(float(idx)/len(words) * 10)
        previous_proportion = int(float(idx - 1)/len(words) * 10)
        if current_proportion != previous_proportion:
            LOG.info(current_proportion)

        tree = build_constituent_tree(word, synt)[0]
        hf.run(tree)
        indices = get_target_indices(target)

        mat = [["*" for _ in range(len(indices) + 1)] for _ in range(len(word))]
        for j in range(len(word)):
            mat[j][0] = "-"
        # print mat
        for i, predicate_index in enumerate(indices):
            predicate = locate_predicate(tree, predicate_index)
            if predicate is None:
                LOG.info("warn: cannot locate predicate, #%d" % idx)
                continue

            cons = generate_candidate_constituents(tree, predicate)
            if len(cons) > 0:
                X = [extract_feature(predicate, con) for con in cons] 
                X1 = attrs.transform([{xx: 1 for xx in x} for x in X])
                Z1 = model_phase_one.predict(X1)
            else:
                LOG.info("warn: no candidates was found.")
                Z1 = []

            mat[predicate_index][0] = target[predicate_index]
            mat[predicate_index][i+1] = "(V*)"
            for j, z in enumerate(Z1):
                if z == 0:
                    continue

                X2 = attrs.transform([{x: 1 for x in X[j]}])
                Z2 = model_phase_two.predict(X2)
                start, end = cons[j].start, cons[j].end
                mat[start][i + 1] = ("(%s" % inversed_labels[Z2[0]]) + mat[start][i+1]
                mat[end - 1][i + 1] += ")"

        print("\n".join(["\t".join(mat[i]) for i in range(len(mat))]))
        print()
    LOG.info("[done]")

if __name__ == "__main__":
    usage = "An implementation of Xue and Palmer (2004): Calibrating Features for Semantic Role Labeling\n"
    usage += ("usage: %s [learn|labeling|validation] [options]" % sys.argv[0])

    if len(sys.argv) < 2:
        print(usage, file=sys.stderr)
        sys.exit(1)
    elif sys.argv[1] == "learn":
        learn()
    elif sys.argv[1] == "labeling":
        labeling()
    elif sys.argv[1] == "validation":
        validation()
    else:
        LOG.info("error: unknown target [%s]" % sys.argv[1])
        sys.exit(1)
