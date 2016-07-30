#!/usr/bin/env python
import sys
import argparse
import random
import logging
import cPickle as pkl
from corpus import read_dataset, get_alphabet
from tb_parser import State, Parser
from model import Model
from tree_utils import is_projective, is_tree


def evaluate(dataset, parser, model):
    n_uas, n_total = 0, 0
    for data in dataset:
        d = [parser.ROOT] + data
        s = State(d)
        while not s.terminate():
            x = parser.extract_features(s)
            x = parser.parameterize_X(x, s)
            best, best_action = None, None
            for i, p in enumerate(model.classify(x)):
                action = parser.get_action(i)
                if s.valid(action) and (best is None or p > best):
                    best = p
                    best_action = action
            s.transit(best_action)

        for i in range(1, len(d)):
            if d[i]['head'] == s.result[i]['head']:
                n_uas += 1
        n_total += (len(d) - 1)
    return float(n_uas) / n_total


def learn():
    conf = argparse.ArgumentParser("Learning component for chen and manning (2014)'s parser")
    conf.add_argument("--model", help="The path to the model.")
    conf.add_argument("--embedding", help="The path to the embedding file.")
    conf.add_argument("--reference", help="The path to the reference file.")
    conf.add_argument("--development", help="The path to the development file.")
    conf.add_argument("--init-range", dest="init_range", type=float, default=0.01,
                      help="The initialization range.")
    conf.add_argument("--max-iter", dest="max_iter",type=int, default=20000,
                      help="The number of max iteration.")
    conf.add_argument("--hidden-size", dest="hidden_size", type=int, default=200, help="The size of hidden layer.")
    conf.add_argument("--embedding-size", dest="embedding_size", type=int, default=50, help="The size of embedding.")
    conf.add_argument("--precomputed-number", dest="precomputed_number", type=int, default=100000,
                      help="The number of precomputed.")
    conf.add_argument("--evaluate-stops", dest="evaluate_stops", type=int, default=100,
                      help="Evaluation on per-iteration.")
    conf.add_argument("--ada-eps", dest="ada_eps", type=float, default=1e-6, help="The EPS in AdaGrad.")
    conf.add_argument("--ada-alpha", dest="ada_alpha", type=float, default=0.01, help="The Alpha in AdaGrad.")
    conf.add_argument("--lambda", dest="lambda", type=float, default=1e-8, help="The regularizer parameter.")
    conf.add_argument("--dropout", dest="dropout", type=float, default=0.5, help="The probability for dropout.")
    opts = conf.parse_args(sys.argv[2:])

    train_dataset = read_dataset(opts.reference)
    devel_dataset = read_dataset(opts.development)
    form_alphabet = get_alphabet(train_dataset, 'form')
    pos_alphabet = get_alphabet(train_dataset, 'pos')
    deprel_alphabet = get_alphabet(train_dataset, 'deprel')
    parser = Parser(form_alphabet, pos_alphabet, deprel_alphabet)
    model = Model(form_size=len(form_alphabet),
                  form_dim=50,
                  pos_size=len(pos_alphabet),
                  pos_dim=50,
                  deprel_size=len(deprel_alphabet),
                  deprel_dim=50,
                  hidden_dim=opts.hidden_size,
                  output_dim=((len(deprel_alphabet) - 2) * 2 + 1)
                  )

    n_sentence = 0
    best_uas = 0.
    for iter in range(1, opts.max_iter + 1):
        logging.info(('Iteration %d' % iter))
        order = range(len(train_dataset))
        random.shuffle(order)
        for n in order:
            train_data = train_dataset[n]
            if not is_tree(train_data) or not is_projective(train_data):
                logging.info('%d sentence is not projective, skipped.' % n)
            X, Y = parser.generate_training_instance(train_data)
            model.train(X, Y)

            n_sentence += 1
            if n_sentence % opts.evaluate_stops == 0:
                uas = evaluate(devel_dataset, parser, model)
                logging.info('Devel at %d, UAS=%f' % (n_sentence, uas))
                if uas > best_uas:
                    best_uas = uas
                    pkl.dump((parser, model), opts.model)
        uas = evaluate(devel_dataset, parser, model)
        logging.info('Devel at the end of iteration %d, UAS=%f' % (iter, uas))
        if uas > best_uas:
            best_uas = uas
            pkl.dump((parser, model), opts.model)


def test():
    conf = argparse.ArgumentParser("Testing component for chen and manning (2014)'s parser")
    conf.add_argument("--model", help="The path to the model.")
    conf.add_argument("--input", help="The path to the embedding file.")
    conf.add_argument("--output", help="The path to the output file.")
    opts = conf.parse_args(sys.argv[2:])

    model, parser = None, None
    test_dataset = read_dataset(opts.input)
    evaluate(test_dataset, parser, model)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "learn":
        learn()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        print ("usage %s [learn|test] option" % sys.argv[0])