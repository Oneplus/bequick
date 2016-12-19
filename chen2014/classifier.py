#!/usr/bin/env python
import argparse
import logging
import numpy as np
import tensorflow as tf
try:
    import bequick
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from bequick.corpus import read_conllx_dataset, get_alphabet
from bequick.embedding import load_embedding
try:
    from .tb_parser import Parser
    from .model import Classifier, initialize_word_embeddings
    from .tree_utils import is_projective, is_tree
    from .evaluate import evaluate
except (ValueError, SystemError) as e:
    from tb_parser import Parser
    from model import Classifier, initialize_word_embeddings
    from tree_utils import is_projective, is_tree
    from evaluate import evaluate

np.random.seed(1234)
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


def main():
    cmd = argparse.ArgumentParser("An implementation of Chen and Manning (2014)'s parser")
    cmd.add_argument("--embedding", help="The path to the embedding file.")
    cmd.add_argument("--reference", help="The path to the reference file.")
    cmd.add_argument("--development", help="The path to the development file.")
    cmd.add_argument("--test", help="The path to the test file.")
    cmd.add_argument("--init-range", dest="init_range", type=float, default=0.01, help="The initialization range.")
    cmd.add_argument("--max-iter", dest="max_iter", type=int, default=10, help="The number of max iteration.")
    cmd.add_argument("--hidden-size", dest="hidden_size", type=int, default=400, help="The size of hidden layer.")
    cmd.add_argument("--embedding-size", dest="embedding_size", type=int, default=100, help="The size of embedding.")
    cmd.add_argument("--evaluate-stops", dest="evaluate_stops", type=int, default=-1, help="Evaluate on per n iters.")
    cmd.add_argument("--ada-eps", dest="ada_eps", type=float, default=1e-6, help="The EPS in AdaGrad.")
    cmd.add_argument("--ada-alpha", dest="ada_alpha", type=float, default=0.01, help="The Alpha in AdaGrad.")
    cmd.add_argument("--lambda", dest="lamb", type=float, default=1e-8, help="The regularizer parameter.")
    cmd.add_argument("--batch-size", dest="batch_size", type=int, default=5000, help="The size of batch.")
    cmd.add_argument("--dropout", dest="dropout", type=float, default=0.5, help="The probability for dropout.")
    opts = cmd.parse_args()

    train_dataset = read_conllx_dataset(opts.reference)
    logging.info("Loaded {0} training sentences.".format(len(train_dataset)))
    devel_dataset = read_conllx_dataset(opts.development)
    logging.info("Loaded {0} development sentences.".format(len(devel_dataset)))
    test_dataset = read_conllx_dataset(opts.test)
    logging.info("Loaded {0} development sentences.".format(len(test_dataset)))

    form_alphabet = get_alphabet(train_dataset, 'form')
    logging.info("# {0} forms in alphabet".format(len(form_alphabet)))
    pos_alphabet = get_alphabet(train_dataset, 'pos')
    logging.info("# {0} postags in alphabet".format(len(pos_alphabet)))
    deprel_alphabet = get_alphabet(train_dataset, 'deprel')
    logging.info("# {0} deprel in alphabet".format(len(deprel_alphabet)))

    parser = Parser(form_alphabet, pos_alphabet, deprel_alphabet)
    model = Classifier(form_size=len(form_alphabet), form_dim=100, pos_size=len(pos_alphabet), pos_dim=20,
                       deprel_size=len(deprel_alphabet), deprel_dim=20, hidden_dim=opts.hidden_size,
                       output_dim=parser.num_actions(), dropout=opts.dropout, l2=opts.lamb)
    indices, matrix = load_embedding(opts.embedding, form_alphabet, opts.embedding_size)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    initialize_word_embeddings(session, model.form_emb, indices, matrix)
    logging.info('Embedding is loaded.')

    best_uas = 0.
    forms, postags, deprels, Ys = [], [], [], []
    for n, train_data in enumerate(train_dataset):
        if not is_tree(train_data):
            logging.info('{0} sentence is not tree, skipped.'.format(n))
            continue
        if not is_projective(train_data):
            logging.info('{0} sentence is not projective, skipped.'.format(n))
            continue
        xs, ys = parser.generate_training_instance(train_data)
        forms.append(xs[0])
        postags.append(xs[1])
        deprels.append(xs[2])
        Ys.append(ys)
    forms = np.concatenate(forms)
    postags = np.concatenate(postags)
    deprels = np.concatenate(deprels)
    Ys = np.concatenate(Ys)

    n_batch, n_samples = 0, Ys.shape[0]
    order = np.arange(n_samples)
    logging.info('Training sample size: {0}'.format(n_samples))
    for i in range(1, opts.max_iter + 1):
        np.random.shuffle(order)
        cost = 0.
        for batch_start in range(0, n_samples, opts.batch_size):
            batch_end = batch_start + opts.batch_size if batch_start + opts.batch_size < n_samples else n_samples
            batch_id = order[batch_start: batch_end]
            xs, ys = (forms[batch_id], postags[batch_id], deprels[batch_id]), Ys[batch_id]
            cost += model.train(session, xs, ys)
            n_batch += 1
            if opts.evaluate_stops > 0 and n_batch % opts.evaluate_stops == 0:
                uas = evaluate(devel_dataset, session, parser, model)
                logging.info('At {0}, UAS={1}'.format((float(n_batch) / n_samples), uas))
                if uas > best_uas:
                    best_uas = uas
                    uas = evaluate(test_dataset, session, parser, model)
                    logging.info('New best achieved: {0}, test: {1}'.format(best_uas, uas))
        uas = evaluate(devel_dataset, session, parser, model)
        logging.info('Iteration {0} done, Cost={1}, UAS={2}'.format(i, cost, uas))
        if uas > best_uas:
            best_uas = uas
            uas = evaluate(test_dataset, session, parser, model)
            logging.info('New best achieved: {0}, test: {1}'.format(best_uas, uas))
    logging.info('Finish training, best UAS: {0}'.format(best_uas))

if __name__ == "__main__":
    main()
