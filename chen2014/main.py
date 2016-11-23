#!/usr/bin/env python
import sys
import argparse
import logging
import numpy as np
from corpus import read_dataset, get_alphabet
from tb_parser import State, Parser
from model import Model
from tree_utils import is_projective, is_tree

np.random.seed(1234)

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


def load_embedding(path, form_alphabet, dim):
    indices = []
    matrix = np.zeros(shape=(len(form_alphabet), dim))
    row = 0
    for line in open(path, 'r'):
        tokens = line.strip().split()
        word = tokens[0]
        if word in form_alphabet:
            key = form_alphabet.get(word)
            indices.append(key)
            matrix[row,:]= np.array([float(x) for x in tokens[1:]])
            row += 1
    return indices, matrix[:row, :]


def evaluate(dataset, parser, model):
    n_uas, n_total = 0, 0
    for data in dataset:
        d = [parser.ROOT] + data
        s = State(d)
        while not s.terminate():
            ctx = parser.extract_features(s)
            x = parser.parameterize_X([ctx], s)
            best, best_action = None, None
            prediction = model.classify(x)[0]
            for i, p in enumerate(prediction):
                action = parser.get_action(i)
                if s.valid(action) and (best is None or p > best):
                    best = p
                    best_action = action
            s.transit(best_action)

        for i in range(1, len(d)):
            if d[i]['head'] == s.result[i]['head']:
                n_uas += 1
            n_total += 1
    return float(n_uas) / n_total


def learn():
    cmd = argparse.ArgumentParser("Learning component for chen and manning (2014)'s parser")
    cmd.add_argument("--model", help="The path to the model.")
    cmd.add_argument("--embedding", help="The path to the embedding file.")
    cmd.add_argument("--reference", help="The path to the reference file.")
    cmd.add_argument("--development", help="The path to the development file.")
    cmd.add_argument("--init-range", dest="init_range", type=float, default=0.01, help="The initialization range.")
    cmd.add_argument("--max-iter", dest="max_iter", type=int, default=10, help="The number of max iteration.")
    cmd.add_argument("--hidden-size", dest="hidden_size", type=int, default=200, help="The size of hidden layer.")
    cmd.add_argument("--embedding-size", dest="embedding_size", type=int, default=100, help="The size of embedding.")
    cmd.add_argument("--evaluate-stops", dest="evaluate_stops", type=int, default=-1, help="Evaluate on per n iters.")
    cmd.add_argument("--ada-eps", dest="ada_eps", type=float, default=1e-6, help="The EPS in AdaGrad.")
    cmd.add_argument("--ada-alpha", dest="ada_alpha", type=float, default=0.01, help="The Alpha in AdaGrad.")
    cmd.add_argument("--lambda", dest="lamb", type=float, default=1e-8, help="The regularizer parameter.")
    cmd.add_argument("--batch-size", dest="batch_size", type=int, default=5000, help="The size of batch.")
    cmd.add_argument("--dropout", dest="dropout", type=float, default=0.5, help="The probability for dropout.")
    opts = cmd.parse_args(sys.argv[2:])

    train_dataset = read_dataset(opts.reference)
    logging.info("Loaded %d training sentences." % len(train_dataset))
    
    devel_dataset = read_dataset(opts.development)
    logging.info("Loaded %d development sentences." % len(devel_dataset))

    form_alphabet = get_alphabet(train_dataset, 'form')
    logging.info("# %d forms in alphabet" % len(form_alphabet))

    pos_alphabet = get_alphabet(train_dataset, 'pos')
    logging.info("# %d postags in alphabet" % len(pos_alphabet))
    
    deprel_alphabet = get_alphabet(train_dataset, 'deprel')
    logging.info("# %d deprel in alphabet" % len(deprel_alphabet))

    parser = Parser(form_alphabet, pos_alphabet, deprel_alphabet)
    model = Model(form_size=len(form_alphabet),
                  form_dim=100,
                  pos_size=len(pos_alphabet),
                  pos_dim=20,
                  deprel_size=len(deprel_alphabet),
                  deprel_dim=20,
                  hidden_dim=opts.hidden_size,
                  output_dim=parser.num_actions(),
                  lambda_=opts.lamb
                  )
    indices, matrix = load_embedding(opts.embedding, form_alphabet, opts.embedding_size)
    model.initialize_word_embeddings(indices, matrix)
    logging.info('Embedding is loaded.')

    best_uas = 0.
    train_samples = []
    for train_data in train_dataset:
        if not is_tree(train_data):
            logging.info('%d sentence is not tree, skipped.' % n)
            continue
        if not is_projective(train_data):
            logging.info('%d sentence is not projective, skipped.' % n)
            continue
        X, Y = parser.generate_training_instance(train_data)
        train_samples.extend(zip(X, Y))

    n_batch, n_samples = 0, len(train_samples)
    for i in range(1, opts.max_iter + 1):
        logging.info(('Iteration %d' % i))
        np.random.shuffle(train_samples)
        cost = 0.
        for s in range(0, n_samples, opts.batch_size):
            batch = train_samples[s: s + opts.batch_size]
            cost += model.train([x for x, y in batch], [y for x, y in batch])
            n_batch += 1
            if opts.evaluate_stops > 0 and n_batch % opts.evaluate_stops == 0:
                uas = evaluate(devel_dataset, parser, model)
                logging.info('At {0}, UAS={1}'.format((float(n_batch) / n_samples), uas))
                if uas > best_uas:
                    best_uas = uas
                    logging.info('New best achieved {0}'.format(best_uas))
        uas = evaluate(devel_dataset, parser, model)
        logging.info('End of iteration {0}, Cost={1}, UAS={2}'.format(i, cost, uas))
        if uas > best_uas:
            best_uas = uas
            logging.info('New best achieved: {0}'.format(best_uas))
    logging.info('Finish training, best uas is {0}'.format(best_uas))


def test():
    cmd = argparse.ArgumentParser("Testing component for chen and manning (2014)'s parser")
    cmd.add_argument("--model", help="The path to the model.")
    cmd.add_argument("--input", help="The path to the embedding file.")
    cmd.add_argument("--output", help="The path to the output file.")
    opts = cmd.parse_args(sys.argv[2:])

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
