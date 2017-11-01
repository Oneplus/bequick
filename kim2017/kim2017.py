#!/usr/bin/env python
from __future__ import print_function
import sys
import argparse


def main():
    cmd = argparse.ArgumentParser()
    cmd.add_argument('--data_file', default='data/sst.train',
                     help='Path to the training file.')
    cmd.add_argument('--val_data_file', default='data/sst.validation',
                     help='Path to the validation file.')
    cmd.add_argument('--num_layers', default=2,
                     help='number of layers in LSTM.')
    cmd.add_argument('--rnn_size', default=500,
                     help='The size of LSTM hidden size.')
    cmd.add_argument('--word_vec_size', default=500,
                     help='The size of word vector.')
    cmd.add_argument('--attention', default='softmax',
                     help='The attention type [softmax, sigmoid, crf]')
    cmd.add_argument('--lambda', default=2,
                     help='Normalization lambda for marginals if using structured attention.')
    cmd.add_argument('--lambda2', default=0.005,
                     help='l2 penalty for CRF bias for structured attention.')

    cmd.add_argument('--epoch', default=30,
                     help='the number of training epochs.')
    cmd.add_argument('--param_init', default=0.1,
                     help='parameters are initialized with uniform distribution (-param_init, +param_init).')
    cmd.add_argument('--optim', default='sgd',
                     help='available optimizer includes: sgd, adagrad, adadelta, adam')
    cmd.add_argument('--learning_rate', default=1, type=float,
                     help='the initial learning rate, recommendation: sgd=1, adagrad=0.1, adadelta=1, adam=1')
    cmd.add_argument('--max_gradient_norm', default=1, type=float,
                     help='If the norm of gradient exceeds this value, renormalize it.')
    cmd.add_argument('--dropout', default=0.3, type=float,
                     help='The dropout rate on LSTM.')


if __name__ == "__main__":
    main()
