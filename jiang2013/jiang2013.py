#!/usr/bin/env python
__author__ = "Yijia Liu"
__email__ = "yijia_liu@sutd.edu.sg"

import sys
import math
import cPickle as pickle
from optparse import OptionParser
from numpy import array, zeros, exp, add, subtract, sum, newaxis

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
        items.append( (tokens[:-1], tokens[-1].split("|")) )
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
        for attrs, labels in instance.items:
            for attr in attrs:
                if attr not in attrs_alphabet:
                    attrs_alphabet[attr] = len(attrs_alphabet)
            for label in labels:
                if label not in labels_alphabet:
                    labels_alphabet[label] = len(labels_alphabet)
    T = len(labels_alphabet)
    for key, value in labels_alphabet.iteritems():
        inverse_labels_alphabet[value] = key

    for instance in instances:
        L = len(instance.items)
        instance.M = M = zeros((L, T), dtype=bool)
        for i in xrange(L):
            attrs, labels = instance.items[i]
            for o in xrange(T):
                if inverse_labels_alphabet[o] in labels:
                    M[i,o] = True
                else:
                    M[i,o] = False
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


def init_learn_opt():
    '''
    Initialize the learning options

    Returns
    -------
    parser : OptionParser
        The option parser for learning
    '''
    usage = "Learning component for Semi-CRF with partial annotated data"
    cmd = OptionParser(usage)
    cmd.add_option("-t", "--train", dest="train", help="the training data.")
    cmd.add_option("-d", "--devel", dest="devel", help="the develop data.")
    cmd.add_option("-s", "--split", dest="split", type=int,
            help="the develop data.")
    cmd.add_option("-m", "--model", dest="model", help="the model file path.")
    cmd.add_option("-i", "--iteration", dest="iteration", default=10, type=int,
            help="specify number of cores to use")
    return cmd


def init_tag_opt():
    '''
    Initialize the learning options

    Returns
    -------
    parser : OptionParser
        The option parser for learning
    '''
    usage = "Tagging component for Semi-CRF with partial annotated data"
    cmd = OptionParser(usage)
    cmd.add_option("-d", "--devel", dest="devel", help="the devel data.")
    cmd.add_option("-m", "--model", dest="model", help="the model file path.")
    return cmd


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


def argmax(g0, g, L, T, init, trans):
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

    s[0] = g0 * init
    p[0] = array([-1] * T)

    for i in range(1, L):
        for t in range(T):
            C = (s[i-1,:] + g[i,:,t]) * trans[:,t]
            s[i,t] = C.max()
            p[i,t] = C.argmax()

    return s, p


def partial_argmax(g0, g, M, L, T, init, trans):
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

    s[0] = g0 * init
    p[0] = array([-1] * T)

    for i in range(1, L):
        for o in range(T):
            if not M[i,o]:
                continue
            for t in range(T):
                if trans[t,o] == 0 or not M[i-1,t]:
                    continue
                if s[i-1,t] + g[i,t,o] > s[i,o]:
                    s[i,o] = s[i-1,t] + g[i,t,o]
                    p[i,o] = t
    return s, p


def partial_viterbi(model, instance, init, trans):
    '''
    '''
    L = len(instance.items)
    M = instance.M
    T = len(model.labels_alphabet)
    A = len(model.attrs_alphabet)

    build_instance(model.attrs_alphabet, model.labels_alphabet, instance)
    g0, g = build_score_cache(model.w, L, T, A, instance)

    s, p = partial_argmax(g0, g, M, L, T, init, trans)
    v, i = s[L -1].argmax(), L -1

    ret = []
    while i >= 0:
        ret.append(v)
        v = p[i][v]
        i -= 1

    ret.reverse()
    return ret



def viterbi(model, instance, init, trans):
    '''
    The viterbi decoding process

    Parameters
    ----------
    model : CRF
        The CRF model
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
    g0, g = build_score_cache(model.w, L, T, A, instance)

    s, p = argmax(g0, g, L, T, init, trans)
    v, i = s[L -1].argmax(), L -1

    ret = []
    while i >= 0:
        ret.append(v)
        v = p[i][v]
        i -= 1

    ret.reverse()
    return ret


def legal_matrix(labels_alphabet):
    legal_init = [0. for i in xrange(len(labels_alphabet))]
    legal_trans = [[0. for i in xrange(4)] for j in xrange(4)]
    LB = labels_alphabet
    legal_init[ LB["B"] ] = 1.
    legal_init[ LB["S"] ] = 1.
    legal_trans[ LB["B"] ][ LB["M"] ] = 1.
    legal_trans[ LB["B"] ][ LB["E"] ] = 1.
    legal_trans[ LB["M"] ][ LB["M"] ] = 1.
    legal_trans[ LB["M"] ][ LB["E"] ] = 1.
    legal_trans[ LB["E"] ][ LB["B"] ] = 1.
    legal_trans[ LB["E"] ][ LB["S"] ] = 1.
    legal_trans[ LB["S"] ][ LB["S"] ] = 1.
    legal_trans[ LB["S"] ][ LB["B"] ] = 1.
    legal_init = array(legal_init)
    legal_trans = array(legal_trans)

    return legal_init, legal_trans


def _tags2words(tags):
    L = len(tags)
    word = "0"
    words = []
    i = 1
    while i < L:
        if tags[i] == "B" or tags[i] == "S":
            words.append(word)
            word = "%d" % i
        else:
            word += "%d" % i
        i += 1
    words.append(word)
    return words


def _eval_on_word(ref_words, ans_words):
    m, n = 0, 0
    ref_len, ans_len = 0, 0
    corr_words = 0
    while m < len(ans_words) and n < len(ref_words):
        if ans_words[m] == ref_words[n]:
            corr_words += 1
            ref_len += len(ref_words[n])
            ans_len += len(ans_words[m])
            m += 1
            n += 1
        else:
            ref_len += len(ref_words[n])
            ans_len += len(ans_words[m])
            m += 1
            n += 1
            while (m < len(ans_words)) and (n < len(ref_words)):
                if (ref_len > ans_len):
                    ans_len += len(ans_words[m])
                    m += 1
                elif (ref_len < ans_len):
                    ref_len += len(ref_words[n])
                    n += 1
                else:
                    break

    return (corr_words, len(ans_words), len(ref_words))


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
        fp=open(opts.devel, "r")
    except:
        print >> sys.stderr, "Failed to open test file."
        sys.exit(1)

    instances = [read_instance(context) for context in fp.read().strip().split("\n\n")]

    keys = ["" for _ in xrange(len(model.labels_alphabet))]
    for k, v in model.labels_alphabet.iteritems():
        keys[v] = k

    legal_init, legal_trans = legal_matrix(model.labels_alphabet)

    for instance in instances:
        predict = viterbi(model, instance, legal_init, legal_trans)
        for index, item in enumerate(instance.items):
            attrs, label = item
            print "%s" % keys[predict[index]]
        print


def evaluate(opts, model):
    try:
        fp=open(opts.devel, "r")
    except:
        print >> sys.stderr, "WARN: Failed to open test file."
        return

    instances = [read_instance(context) for context in fp.read().strip().split("\n\n")]

    keys = ["" for _ in xrange(len(model.labels_alphabet))]
    for k, v in model.labels_alphabet.iteritems():
        keys[v] = k

    legal_init, legal_trans = legal_matrix(model.labels_alphabet)

    corr_words, ans_words, ref_words = 0, 0, 0
    for instance in instances:
        answer = [keys[t] for t in viterbi(model, instance, legal_init, legal_trans)]
        reference = [label[0] for attr, label in instance.items]
        res = _eval_on_word(reference, answer)

        corr_words += res[0]
        ans_words += res[1]
        ref_words += res[2]

    p = float(corr_words)/ans_words*100
    r = float(corr_words)/ref_words*100
    f = 0 if p+r == 0 else p*r*2/(p+r)
    print >> sys.stderr, "p=%f r=%f f=%f" % (p,r,f)


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
        fp=open(opts.train, "r")
    except:
        print >> sys.stderr, "Failed to open file."
        sys.exit(1)

    # read instances
    instances = [read_instance(instance) for instance in fp.read().strip().split("\n\n")]
    print >> sys.stderr, "# of instances : %d" % len(instances)
    # build feature space
    attrs_alphabet, labels_alphabet = build_feature_space(instances)
    print >> sys.stderr, "# of attributes : %d" % len(attrs_alphabet)
    print >> sys.stderr, "# of labels : %d" % len(labels_alphabet)

    # build instances
    build_instances(attrs_alphabet, labels_alphabet, instances)

    # initialize the parameters
    T, A = len(labels_alphabet), len(attrs_alphabet)
    w = zeros((T + A) * T, dtype=float)
    wsum = zeros((T+A)*T, dtype=float)
    wtime= zeros((T+A)*T, dtype=int)
    print >> sys.stderr, "Size of feature space is %d" % w.shape[0]

    legal_init, legal_trans = legal_matrix(labels_alphabet)

    model = Meta()
    model.labels_alphabet = labels_alphabet
    model.attrs_alphabet  = attrs_alphabet
    model.w = w

    keys = ["" for _ in xrange(len(model.labels_alphabet))]
    for k, v in model.labels_alphabet.iteritems():
        keys[v] = k

    N = len(instances[:opts.split])
    print >> sys.stderr, "Pharse 1: Train on %d data" % N
    # Pharse 1, train the baseline model
    for iteration in xrange(opts.iteration):
        print >> sys.stderr, "Pharse 1: Iteration %d, start" % iteration
        for i, instance in enumerate(instances[:opts.split]):
            hoc=(i+1)*19/N
            if hoc > i*19/N:
                print >> sys.stderr, ("%d"%(hoc/2+1) if hoc&1 else "."),
            answer = viterbi(model, instance, legal_init, legal_trans)
            reference = [labels_alphabet[label[0]] for attr, label in instance.items]

            U = instance.unigram_feature_table
            B = instance.bigram_feature_table
            # assert(len(predict) == len(answers))
            now=iteration*N+i
            for index, (answ, refer) in enumerate(zip(answer, reference)):
                if answ != refer:
                    update(w, wsum, wtime, U[index, refer], now, 1)
                    update(w, wsum, wtime, U[index, answ], now, -1)
            for j in xrange(len(answer) - 1):
                if answer[j] != reference[j] or answer[j+1] != reference[j+1]:
                    update(w, wsum, wtime, B[reference[j], reference[j+1]], now, 1)
                    update(w, wsum, wtime, B[answer[j], answer[j+1]], now, -1)
        flush(w, wsum, wtime, (iteration+1)*N)
        print >> sys.stderr, "Pharse 1: Iteration %d, done" % iteration
        model.w = wsum
        evaluate(opts, model)
        model.w = w

    grand = opts.iteration*N
    N = len(instances[opts.split:])
    print >> sys.stderr, "Pharse 2: Train on %d data" % N
    # Pharse 2, train with constraints
    for iteration in xrange(opts.iteration):
        print >> sys.stderr, "Pharse 2: Iteration %d, start" % iteration
        for i, instance in enumerate(instances[opts.split:]):
            hoc=(i+1)*19/N
            if hoc > i*19/N:
                print >> sys.stderr, ("%d"%(hoc/2+1) if hoc&1 else "."),
            answer = viterbi(model, instance, legal_init, legal_trans)
            reference = partial_viterbi(model, instance, legal_init, legal_trans)

            U = instance.unigram_feature_table
            B = instance.bigram_feature_table
            # assert(len(predict) == len(answers))
            now=iteration*N+i+grand
            for index, (answ, refer) in enumerate(zip(answer, reference)):
                if answ != refer:
                    update(w, wsum, wtime, U[index, refer], now, 1)
                    update(w, wsum, wtime, U[index, answ], now, -1)
            for j in xrange(len(answer) - 1):
                if answer[j] != reference[j] or answer[j+1] != reference[j+1]:
                    update(w, wsum, wtime, B[reference[j], reference[j+1]], now, 1)
                    update(w, wsum, wtime, B[answer[j], answer[j+1]], now, -1)
        flush(w, wsum, wtime, (iteration+1)*N+grand)
        print >> sys.stderr, "Pharse 2: Iteration %d, done" % iteration
        model.w = wsum
        evaluate(opts, model)
        model.w = w


    try:
        fpo=open(opts.model, "wb")
    except:
        print >> sys.stderr, "Failed to open model file."
        sys.exit(1)

    pickle.dump(model, fpo)



if __name__=="__main__":
    usage = "Semi-CRF with partial annotated data\n"
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
