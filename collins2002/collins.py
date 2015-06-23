#!/usr/bin/env python
__author__ = "Yijia Liu"
__email__ = "oneplus.lau@gmail.com"
import gzip
import sys
import math
import random
import cPickle as pickle
import gc

from datetime import datetime
from optparse import OptionParser

from numpy import array, zeros, sum
from numpy import linalg

from scipy.sparse import dok_matrix

gc.disable()

class Meta(object):
    pass


def read_instance(context):
    '''
    Read one instance from context blocks, ONLY SUPPORT
    mallet format.

    Parameters
    ----------
    context : str
        The instance context
    '''
    instance = Meta()
    instance.items = items = []
    for line in context.split("\n"):
        tokens = line.split()
        items.append( (tokens[1:], tokens[0]) )
    return instance


def build_feature_space(instances):
    '''
    Build feature space from instances

    Parameters
    ----------
    instances : list(Instance)
        the collection of instances, each instance is a list of tuple
        of (attrs, label)

    Returns
    -------
    attrs : dict
        the dict of attributes
    labels : dict
        the dict of labels
    '''
    attrs_alphabet = {}
    labels_alphabet = {}
    inverse_labels_alphabet = {}

    for instance in instances:
        for attrs, label in instance.items:
            for attr in attrs:
                if attr not in attrs_alphabet:
                    attrs_alphabet[attr] = len(attrs_alphabet)
            if label not in labels_alphabet:
                labels_alphabet[label] = len(labels_alphabet)
    T = len(labels_alphabet)
    for key, value in labels_alphabet.iteritems():
        inverse_labels_alphabet[value] = key

    return attrs_alphabet, labels_alphabet


def build_instance(attrs_alphabet, labels_alphabet, instance):
    '''
    Build the instance from raw instance

    Parameters
    ----------
    attrs_alphabet : dict
        The dict of attributes
    labels_alphabet : dict
        The dict of labels
    instance : Instance
        The instance
    '''
    if (hasattr(instance, 'unigram_feature_table') and
            hasattr(instance, 'bigram_feature_table')):
        return

    instance.unigram_feature_table = U = {}
    instance.bigram_feature_table = B = {}

    T = len(labels_alphabet)
    A = len(attrs_alphabet)
    for i, item in enumerate(instance.items):
        attrs, labels = item
        attrs = [attrs_alphabet[attr] for attr in attrs if attr in attrs_alphabet]
        for k in xrange(T):
            U[i,k] = array([a * T + k for a in attrs], dtype=int)
    for j in xrange(T):
        for k in xrange(T):
            B[j,k] = array([(A + j) * T + k], dtype=int)


def destroy_instance(instance):
    '''
    Destroy the instance

    Parameters
    ----------
    instance: Instance
        The instance
    '''
    del instance.unigram_feature_table
    del instance.bigram_feature_table


def build_instances(attrs_alphabet, labels_alphabet, instances):
    '''
    Function for building instances

    Parameters
    ----------
    attrs_alphabet : dict
        The dict of attributes
    labels_alphabet : dict
        The dict of labels
    '''
    for instance in instances:
        build_instance(attrs_alphabet, labels_alphabet, instance)



def build_score_cache(w, L, T, A, instance):
    '''
    Build the score cache

    Parameters
    ----------
    w : array
        The parameters
    L : int
        The length of instances
    T : int
        The number of labels
    A : int
        The number of attributes
    instance : Instance
        The instance class
    '''
    g0 = zeros(T, dtype=float)
    g = zeros((L, T, T), dtype=float)

    U = instance.unigram_feature_table
    for j in xrange(T):
        g0[j] = w.take(U[0,j]).sum()

    bundle = [array(range(A*T+_,A*T+T*T+_,T)) for _ in xrange(T)]
    for i in xrange(1, L):
        for k in xrange(T):
            pv = w.take(U[i,k]).sum()
            g[i,:,k] = pv + w.take(bundle[k])
    return (g0, g)


def argmax(g0, g, L, T):
    '''
    The argmax function, used in viterbi decoding

    Parameters
    ----------
    g0 : array
        The initial vector, only with the emission score
    g : array
        The combo matrix, with the emission and transition score
    L : int
        The length of instance
    T : int
        The number of tags
    '''
    s = zeros((L, T), dtype=float)
    p = zeros((L, T), dtype=int)

    s[0] = g0
    p[0] = array([-1] * T)

    for i in range(1, L):
        for t in range(T):
            C = s[i-1,:] + g[i,:,t]
            s[i,t] = C.max()
            p[i,t] = C.argmax()

    return s, p


def viterbi(model, avg, instance):
    '''
    The viterbi decoding process

    Parameters
    ----------
    model : CRF
        The CRF model
    avg : bool
        If use averaged parameters
    instance : Instance
        The instances

    Returns
    -------
    ret : list(int)
        The label sequence.
    '''
    L = len(instance.items)
    T = len(model.labels_alphabet)
    A = len(model.attrs_alphabet)

    build_instance(model.attrs_alphabet, model.labels_alphabet, instance)
    g0, g = build_score_cache(model.wsum if avg else model.w, L, T, A, instance)

    s, p = argmax(g0, g, L, T)
    v, i = s[L -1].argmax(), L -1

    ret = []
    while i >= 0:
        ret.append(v)
        v = p[i][v]
        i -= 1

    ret.reverse()
    return ret


def tag(opts):
    '''
    '''
    try:
        fpm=open(opts.model, "rb")
    except:
        print >> sys.stderr, "Failed to open model file."
        sys.exit(1)
    model = pickle.load(fpm)
    try:
        if opts.devel.endswith('.gz'):
            fp=gzip.open(opts.devel, 'r')
        else:
            fp=open(opts.devel, "r")
    except:
        print >> sys.stderr, "Failed to open test file."
        sys.exit(1)

    instances = [read_instance(context) for context in fp.read().strip().split("\n\n")]

    keys = ["" for _ in xrange(len(model.labels_alphabet))]
    for k, v in model.labels_alphabet.iteritems():
        keys[v] = k

    for instance in instances:
        predict = viterbi(model, True, instance)
        for index, item in enumerate(instance.items):
            attrs, label = item
            print "%s" % keys[predict[index]]
        print


def evaluate(opts, model):
    try:
        if opts.devel.endswith('.gz'):
            fp=gzip.open(opts.devel, "r")
        else:
            fp=open(opts.devel, 'r')
    except:
        print >> sys.stderr, "WARN: Failed to open test file."
        return

    instances = [read_instance(context) for context in fp.read().strip().split("\n\n")]

    keys = ["" for _ in xrange(len(model.labels_alphabet))]
    for k, v in model.labels_alphabet.iteritems():
        keys[v] = k

    corr_tags, total_tags = 0, 0
    for instance in instances:
        answer = [keys[t] for t in viterbi(model, True, instance)]
        reference = [label for attr, label in instance.items]

        for a, r in zip(answer, reference):
            if a == r:
                corr_tags += 1
            total_tags += 1

    p = float(corr_tags)/total_tags*100
    return p


def flush(w, wsum, wtime, now):
    wsum += (now - wtime) * w
    wtime = now


def update(w, wsum, wtime, indices, now, scale):
    elapsed = now-wtime[indices]
    cur_val = w[indices]
    w[indices] = cur_val + scale
    wsum[indices] += (elapsed*cur_val+scale)
    wtime[indices] = now


def learn(opts):
    '''
    Main process for learning parameters from the data

    Parameters
    ----------
    opts : Options
        The options
    '''
    try:
        if opts.train.endswith('.gz'):
            fp=gzip.open(opts.train, 'r')
        else:
            fp=open(opts.train, "r")
    except:
        print >> sys.stderr, "Failed to open file."
        sys.exit(1)

    # read instances
    instances = [read_instance(instance) for instance in fp.read().strip().split("\n\n")]
    print >> sys.stderr, "number of instances : %d" % len(instances)
    # build feature space
    attrs_alphabet, labels_alphabet = build_feature_space(instances)
    print >> sys.stderr, "number of attributes : %d" % len(attrs_alphabet)
    print >> sys.stderr, "number of labels : %d" % len(labels_alphabet)

    # build instances
    # build_instances(attrs_alphabet, labels_alphabet, instances)

    # initialize the parameters
    T, A = len(labels_alphabet), len(attrs_alphabet)
    w = zeros((T + A) * T, dtype=float)
    wsum = zeros((T+A)*T, dtype=float)
    wtime= zeros((T+A)*T, dtype=int)
    print >> sys.stderr, "size of feature space is %d" % w.shape[0]

    model = Meta()
    model.labels_alphabet = labels_alphabet
    model.attrs_alphabet  = attrs_alphabet
    model.w = w
    model.wsum = wsum

    keys = ["" for _ in xrange(len(model.labels_alphabet))]
    for k, v in model.labels_alphabet.iteritems():
        keys[v] = k

    N = len(instances)
    for iteration in xrange(opts.iteration):
        print >> sys.stderr, "iteration %d, start(%s)" % (iteration, 
                datetime.strftime(datetime.now(), "%H:%M:%S")),
        if opts.shuffle:
            random.shuffle(instances)

        for i, instance in enumerate(instances):
            hoc=(i+1)*19/N
            if hoc > i*19/N:
                print >> sys.stderr, ("%d"%(hoc/2+1) if hoc&1 else "."),

            build_instance(attrs_alphabet, labels_alphabet, instance)
            answer = viterbi(model, False, instance)
            reference = [labels_alphabet[label] for attr, label in instance.items]

            U = instance.unigram_feature_table
            B = instance.bigram_feature_table
            # assert(len(predict) == len(answers))
            now=iteration*N+i+1
            if opts.algorithm == "pa":
                s = 0.
                err = 0.
                updates = {}
                for index, (answ, refer) in enumerate(zip(answer, reference)):
                    if answ != refer:
                        s += w.take(U[index, refer]).sum() - w.take(U[index, answ]).sum()
                        err += 1
                        updates.update((u, 1) for u in U[index, refer])
                        updates.update((u, -1) for u in U[index, answ])

                for j in xrange(len(answer) - 1):
                    if answer[j] != reference[j] or answer[j+1] != reference[j+1]:
                        s += w[B[reference[j], reference[j+1]]] - w[B[answer[j], answer[j+1]]]
                        updates.update((b, 1) for b in B[reference[j], reference[j+1]])
                        updates.update((b, -1) for b in B[answer[j], answer[j+1]])

                norm = sum(v*v for k, v in updates.iteritems())
                if norm == 0:
                    step = 0
                else:
                    step = (s + err) / sum(v*v for k, v in updates.iteritems())
            else:
                step = 1

            for index, (answ, refer) in enumerate(zip(answer, reference)):
                if answ != refer:
                    update(w, wsum, wtime, U[index, refer], now, step)
                    update(w, wsum, wtime, U[index, answ], now, -step)
            for j in xrange(len(answer) - 1):
                if answer[j] != reference[j] or answer[j+1] != reference[j+1]:
                    update(w, wsum, wtime, B[reference[j], reference[j+1]], now, step)
                    update(w, wsum, wtime, B[answer[j], answer[j+1]], now, -step)
            if not opts.cached:
                destroy_instance(instance)

        flush(w, wsum, wtime, (iteration+1)*N)
        print >> sys.stderr, ("done(%s)" % datetime.strftime(datetime.now(), "%H:%M:%S")),
        print >> sys.stderr, ("|w| = %f" % linalg.norm(w))
        print >> sys.stderr, "iteration %d, evalute tagging accuracy = %f" % (iteration,
                evaluate(opts, model))

    try:
        fpo=open(opts.model, "wb")
    except:
        print >> sys.stderr, "Failed to open model file."
        sys.exit(1)

    pickle.dump(model, fpo)


def init_learn_opt():
    '''
    Initialize the learning options

    Returns
    -------
    parser : OptionParser
        The option parser for learning
    '''
    usage = "Learning component for Semi-CRF with partial annotated data"
    parser = OptionParser(usage)
    parser.add_option("-t", "--train", dest="train", help="the training data.")
    parser.add_option("-d", "--devel", dest="devel", help="the develop data.")
    parser.add_option("-m", "--model", dest="model", help="the model file path.")
    parser.add_option("-a", "--algorithm", dest="algorithm", help="the learning algorithm.")
    parser.add_option("-s", "--shuffle", dest="shuffle", action="store_true", default=False,
            help="use to specify shuffle the instance.")
    parser.add_option("-c", "--cached", dest="cached", action="store_true", default=False,
            help="cache the data feature, large mem consumption.")
    parser.add_option("-i", "--iteration", dest="iteration", default=10, type=int, 
            help="specify number of iteration")
    return parser


def init_tag_opt():
    '''
    Initialize the learning options

    Returns
    -------
    parser : OptionParser
        The option parser for learning
    '''
    usage = "Tagging component for Semi-CRF with partial annotated data"
    parser = OptionParser(usage)
    parser.add_option("-d", "--devel", dest="devel", help="the devel data.")
    parser.add_option("-m", "--model", dest="model", help="the model file path.")
    return parser


if __name__=="__main__":
    usage = "An implementation of Collins (2002): Discriminative training methods for hidden markov models.\n"
    usage += "usage %s [learn|tag] [options]" % sys.argv[0]

    if len(sys.argv) > 1 and sys.argv[1] == "learn":
        parser = init_learn_opt()
        opts, args = parser.parse_args()
        learn(opts)
    elif len(sys.argv) > 1 and sys.argv[1] == "tag":
        parser = init_tag_opt()
        opts, args = parser.parse_args()
        tag(opts)
    else:
        print >> sys.stderr, usage
        sys.exit(1)

