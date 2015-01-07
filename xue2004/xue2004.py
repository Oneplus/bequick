#!/usr/bin/env python
import sys
import cPickle as pickle
import gzip
from optparse import OptionParser
from xue2004.ioutils import read_syntax_dir, read_props_dir, read_words_dir
from xue2004.props import get_props_indices, get_predicate_arguments, locate_predicate
from xue2004.targets import get_target_indices
from xue2004.head_finder import CollinsHeadFinder
from xue2004.feature import extract_feature
from xue2004.constituent import build_constituent_tree, generate_candidate_constituents
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

__author__="Yijia Liu"
__email__="oneplus.lau@gmail.com"


def init_learn_options():
    usage = "the learning component of xue2004."
    parser = OptionParser(usage)
    parser.add_option("-s", "--synt", dest="synt",
            help="use to specify the syntax file.")
    parser.add_option("-p", "--props", dest="props",
            help="use to specify the prop file.")
    parser.add_option("-w", "--words", dest="words",
            help="use to specify the words file.")
    parser.add_option("-m", "--model", dest="model",
            help="use to specify the model path.")
    return parser

def init_labeling_options():
    usage = "the labeling component of xue2004."
    parser = OptionParser(usage)
    parser.add_option("-s", "--syntax", dest="synt",
            help="use to specify the syntax file.")
    parser.add_option("-t", "--targets", dest="targets",
            help="use to specify the target file.")
    parser.add_option("-w", "--words", dest="words",
            help="use to specify the input words.")
    parser.add_option("-m", "--model", dest="model",
            help="use to specify the model path.")
    return parser


def learn():
    parser = init_learn_options()
    opts, args = parser.parse_args()

    ## Reading the data.
    print >> sys.stderr, "trace: reading data. ",
    words = read_words_dir(opts.words)
    syntax = read_syntax_dir(opts.synt)
    props = read_props_dir(opts.props)
    print >> sys.stderr, "[done]"

    if len(syntax) != len(props) or len(syntax) != len(words):
        print >> sys.stderr, "error: #syntax not equals to #props"
        return

    print >> sys.stderr, "trace: read in %d sentence(s)." % len(words)
    hf = CollinsHeadFinder()
    print >> sys.stderr, "trace: extracting features: 0",
    X, Y = [], []
    for idx, (word, synt, prop) in enumerate(zip(words, syntax, props)):
        current_proportion = int(float(idx)/len(words)* 10)
        previous_proportion = int(float(idx- 1)/len(words)* 10)
        if current_proportion != previous_proportion:
            print >> sys.stderr, current_proportion,

        #if idx != 1:
        #    continue

        tree = build_constituent_tree(word, synt)[0]
        hf.run(tree)
        slices = get_props_indices(prop)

        for predicate_slice in slices:
            predicate = locate_predicate(tree, predicate_slice)
            if predicate is None:
                print >> sys.stderr, ("warn: cannot locate predicate, #%d" % idx)
                continue

            #print >> sys.stderr, predicate
            args = get_predicate_arguments(prop, predicate)
            cons = generate_candidate_constituents(tree, predicate)

            for con in cons:
                found = False
                for key, (start, end) in args.iteritems():
                    if start == con.start and end == con.end:
                        found = True
                        break
                X.append(extract_feature(predicate, con))
                Y.append((1, key) if found else (0, None))
    attrs = DictVectorizer()
    X1 = attrs.fit_transform([{xx:1 for xx in x} for x in X])
    print >> sys.stderr, "[done]"
    print >> sys.stderr, ("trace: extract %d feature(s)" % len(attrs.vocabulary_))
    print >> sys.stderr, ("trace: generate %d training instance(s)." % len(X))
    print >> sys.stderr, ("trace: number of positive instance(s) is %d" % sum([y[0] for y in Y]))
    print >> sys.stderr, "trace: learning phase 1 model.",
    model_phase_one = LogisticRegression(C=1)
    model_phase_one.fit(X1, [y[0] for y in Y])
    print >> sys.stderr, "[done]"

    X2 = attrs.transform([{x:1 for x in X[i]} for i in xrange(len(X)) if Y[i][0] == 1])
    labels = {}
    for y, lab in Y:
        if y == 1 and lab not in labels:
            labels[lab] = len(labels)
    Y2 = [labels[y[1]] for y in Y if y[0] == 1]
    print >> sys.stderr, ("trace: classifing %d types" % len(labels))
    print >> sys.stderr, "trace: learning phase 2 model.",
    model_phase_two = LogisticRegression(C=1)
    model_phase_two.fit(X2, Y2)
    print >> sys.stderr, "[done]"

    pickle.dump((labels, attrs, model_phase_one, model_phase_two),
            gzip.open(opts.model, "w"))

def labeling():
    parser = init_labeling_options()
    opts, args = parser.parse_args()

    print >> sys.stderr, "trace: reading data.",
    words = read_words_dir(opts.words)
    syntax = read_syntax_dir(opts.synt)
    targets = read_props_dir(opts.targets)
    print >> sys.stderr, "[done]"

    if len(syntax) != len(targets) or len(syntax) != len(words):
        print >> sys.stderr, "error: #syntax not equals to #props"
        return

    print >> sys.stderr, "trace: loading model.",
    labels, attrs, model_phase_one, model_phase_two = pickle.load(gzip.open(opts.model, "r"))
    print >> sys.stderr, "[done]"

    inversed_labels = {}
    for label in labels:
        inversed_labels[labels[label]] = label
    hf = CollinsHeadFinder()
    print >> sys.stderr, "trace: predicting: 0",
    for idx, (word, synt, target) in enumerate(zip(words, syntax, targets)):
        current_proportion = int(float(idx)/len(words)* 10)
        previous_proportion = int(float(idx- 1)/len(words)* 10)
        if current_proportion != previous_proportion:
            print >> sys.stderr, current_proportion,
        #if idx != 1:
        #    continue

        tree = build_constituent_tree(word, synt)[0]
        hf.run(tree)
        indices = get_target_indices(target)

        mat = [["*" for i in xrange(len(indices)+ 1)] for j in xrange(len(word))]
        for j in xrange(len(word)): mat[j][0] = "-"
        #print mat
        for i, predicate_index in enumerate(indices):
            predicate = locate_predicate(tree, predicate_index)
            if predicate is None:
                print >> sys.stderr, ("warn: cannot locate predicate, #%d" % idx)
                continue

            #print >> sys.stderr, predicate_index, predicate
            #args = get_predicate_arguments(prop, predicate)
            cons = generate_candidate_constituents(tree, predicate)

            X = [extract_feature(predicate, con) for con in cons] 
            X1 = attrs.transform([{xx:1 for xx in x} for x in X])
            Z1 = model_phase_one.predict(X1)
            #print Z1

            mat[predicate_index][0] = target[predicate_index]
            mat[predicate_index][i+1] = "(V*)"
            for idx, z in enumerate(Z1):
                if z == 0:
                    continue

                X2 = attrs.transform([{x: 1 for x in X[idx]}])
                Z2 = model_phase_two.predict(X2)
                start, end = cons[idx].start, cons[idx].end
                mat[start][i+ 1] = ("(%s" % inversed_labels[Z2[0]]) + mat[start][i+1]
                mat[end- 1][i+ 1] = mat[end-1][i+1] + ")"

        print "\n".join(["\t".join(mat[i]) for i in xrange(len(mat))])
        print

    print >> sys.stderr, "[done]"

if __name__=="__main__":
    usage =  "description: an implementation of Xue and Palmer (2004). \n"
    usage += "link: http://aclweb.org/anthology/W/W04/W04-3212.pdf\n"
    usage +=("usage: %s [learn|labeling] [options]" % sys.argv[0])

    if len(sys.argv) < 2:
        print >> sys.stderr, usage
        sys.exit(1)
    elif sys.argv[1] == "learn":
        learn()
    elif sys.argv[1] == "labeling":
        labeling()
    else:
        print >> sys.stderr, ("error: unknown target [%s]" % sys.argv[1])
        sys.exit(1)
