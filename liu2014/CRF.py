#!/usr/bin/env python
__author__ = "Liu, Yijia"
__email__ = "oneplus.lau@gmail.com"

import math
import sys

from optparse import OptionParser
from scipy.misc import logsumexp
from numpy import array, zeros, exp, add, subtract, sum, newaxis


class Meta(object):
    pass


def logsumexp2(a):
    '''
    Logsumexp
    ---------

    It is a little faster than the scipy.misc.logsumexp

    - param[in] array-like
    '''
    max_element = max(a)
    return max_element + math.log(exp(a - max_element).sum())


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
        items.append( (tokens[1:], tokens[0].split("|")) )
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
    parser = OptionParser(usage)
    parser.add_option("-t", "--train", dest="train", help="the training data.")
    parser.add_option("-m", "--model", dest="model", help="the model file path.")
    parser.add_option("--cores", dest="nr_cores", default=1, type=int, 
                      help="specify number of cores to use")
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


def forward(g0, g, L, T):
    '''
    The forward process

    Parameters
    ----------
    g0 : array
        The initial vector, only with emission score
    g : array
        The combo matrix, with emission and transition score
    L : int
        The length of instance
    T : int
        The number of labels

    Returns
    -------
    a : array
        The matrix for forward process
    '''
    a = zeros((L, T), dtype=float)
    a[0,:] = g0
    for i in xrange(1, L):
        a[i,:] = logsumexp(a[i-1,:,newaxis] + g[i,:,:], axis=0)
    return a


def backward(g, L, T):
    '''
    The backward processs

    Parameters
    ----------
    g : array
    '''
    b = zeros((L, T), dtype=float)
    for i in xrange(L-2, -1, -1):
        b[i,:] = logsumexp(b[i+1,:] + g[i+1,:,], axis=1)
    return b


def partial_forward(g0, g, M, L, T):
    '''
    The partial version of forward algorithm

    Parameters
    ----------
    g0 : array
        The initial score vector
    g : array
        The score matrix
    M : array
        The legal matrix
    L : int
        The length of instance
    T : int
        The number of tags

    Returns
    -------
    a : array
        The result matrix of forward process
    '''
    a = zeros((L, T), dtype=float)
    for o in [_ for _ in xrange(T) if M[0,_]]:
        a[0,o] = g0[o]
    for i in xrange(1, L):
        #a[i,M[i,:]] = logsumexp(g[i,M[i-1,:],:] + a[i-1,M[i-1,:],newaxis], axis=0).take(M[i,:])
        for o in [_ for _ in xrange(T) if M[i,_]]:
            a[i,o] = logsumexp2([a[i-1,_] + g[i,_,o] for _ in xrange(T) if M[i-1,_]])
    return a


def partial_backward(g, M, L, T):
    '''
    The partial version of backward algorithm

    Parameters
    ----------
    g : array
        The score matrix
    M : array
        The legal matrix
    L : int
        The length of instance
    T : int
        The number of tags

    Returns
    -------
    b : array
        The result matrix of backward process
    '''
    b = zeros((L, T), dtype=float)
    for i in xrange(L-2, -1, -1):
        #b[i,M[i,:]] = logsumexp(g[i+1,:,M[i+1,:]].T + b[i+1,M[i+1,:]], axis=1)[M[i,:]]
        for o in [_ for _ in xrange(T) if M[i,_]]:
            b[i,o] = logsumexp2([b[i+1,_] + g[i+1,o,_] for _ in xrange(T) if M[i+1,_]])
    return b


def likelihood_and_dlikelihood(w, T, A, instance):
    '''
    '''
    f, grad = 0, zeros(w.shape[0], dtype=float)

    L = len(instance.items)
    M = instance.M
    g0, g = build_score_cache(w, L, T, A, instance)

    alpha = forward(g0, g, L, T)
    beta  = backward(g, L, T)
    partial_alpha = partial_forward(g0, g, M, L, T)
    partial_beta = partial_backward(g, M, L, T)

    logZ = logsumexp2(alpha[L-1,:])
    partial_logZ = logsumexp2([partial_alpha[L-1,_] for _ in xrange(T) if M[L-1,_]])

    # the likelihood
    f = partial_logZ - logZ

    # the gradient
    U = instance.unigram_feature_table
    B = instance.bigram_feature_table

    # 1st-term
    c = exp(g0 + partial_beta[0,:] - partial_logZ).clip(0.,1.)
    for j in [_ for _ in xrange(T) if M[0,_]]:
        grad[U[0,j]] += c[j]
    for i in xrange(1, L):
        c = exp(add.outer(partial_alpha[i-1,:], partial_beta[i,:]) \
                + g[i,:,:] - partial_logZ).clip(0.,1.)
        for k in [_ for _ in xrange(T) if M[i,_]]:
            for j in [__ for __ in xrange(T) if M[i-1,__]]:
                grad[U[i,k]] += c[j,k]
                grad[B[j,k]] += c[j,k]

    # 2nd-term
    c = exp(g0 + beta[0,:] - logZ).clip(0.,1.)
    for j in xrange(T):
        grad[U[0,j]] -= c[j]
    for i in xrange(1, L):
        c = exp(add.outer(alpha[i-1,:], beta[i,:]) + g[i,:,:] - logZ).clip(0.,1.)
        c_margin = c.sum(axis=0)
        for k in xrange(T):
            grad[U[i,k]] -= c_margin[k]
            #subtract.at(grad, U[i,k], c[:,k].sum())
        grad[range(A*T, (A+T)*T)] -= c.flatten()
    return f, grad


def likelihood_and_dlikelihood_batch(w, T, A, instances, delta=1.):
    '''
    Batchly calculate the likelihood and dlikelihood
    '''
    N = len(instances)
    f, grad = 0., zeros(w.shape[0], dtype=float)
    for index, instance in enumerate(instances):
        if (index + 1) % 1000 == 0:
            print >> sys.stderr, ".",
        if (index + 1) % 10000 == 0:
            print >> sys.stderr
        delta_f, delta_grad = likelihood_and_dlikelihood(w, T, A, instance)
        f += delta_f
        grad += delta_grad
    print >> sys.stderr
    return -(f-((w**2).sum()/(2*delta))), -(grad-(w/delta))


