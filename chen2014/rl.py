#!/usr/bin/env python
import sys
import argparse
import logging
import numpy as np
import tensorflow as tf
from corpus import read_dataset, get_alphabet
from tb_parser import Parser, State
from model import DeepQNetwork, initialize_word_embeddings
from tree_utils import is_projective, is_tree
from evaluate import evaluate
from embedding import load_embedding

np.random.seed(1234)
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


def find_best(parser, state, scores):
    best_score, best_i, best_name = None, None, None
    for i, score in enumerate(scores):
        name = parser.get_action(i)
        if state.valid(name) and (best_score is None or score > best_score):
            best_score, best_i, best_name = score, i, name
    return best_score, best_i, best_name


def get_valid_actions(parser, state):
    n_actions = parser.num_actions()
    aids, names = [], []
    mask = np.zeros(n_actions, dtype=np.bool)
    for aid in range(n_actions):
        name = parser.get_action(aid)
        if state.valid(name):
            aids.append(aid)
            names.append(name)
            mask[aid] = True
    return aids, names, mask


def main():
    cmd = argparse.ArgumentParser("Learning component for chen and manning (2014)'s parser")
    cmd.add_argument("--model", help="The path to the model.")
    cmd.add_argument("--embedding", help="The path to the embedding file.")
    cmd.add_argument("--reference", help="The path to the reference file.")
    cmd.add_argument("--development", help="The path to the development file.")
    cmd.add_argument("--test", help="The path to the test file.")
    cmd.add_argument("--init-range", dest="init_range", type=float, default=0.01, help="The initialization range.")
    cmd.add_argument("--max-iter", dest="max_iter", type=int, default=10, help="The number of max iteration.")
    cmd.add_argument("--hidden-size", dest="hidden_size", type=int, default=400, help="The size of hidden layer.")
    cmd.add_argument("--embedding-size", dest="embedding_size", type=int, default=100, help="The size of embedding.")
    cmd.add_argument("--evaluate-stops", dest="evaluate_stops", type=int, default=-1, help="Evaluate on per n iters.")
    cmd.add_argument("--lambda", dest="lamb", type=float, default=1e-8, help="The regularizer parameter.")
    cmd.add_argument("--dropout", dest="dropout", type=float, default=0.5, help="The probability for dropout.")
    cmd.add_argument("--eps-init", dest="eps_init", type=float, default=1., help="The initial value of eps.")
    cmd.add_argument("--eps-final", dest="eps_final", type=float, default=0.1, help="The final value of eps.")
    cmd.add_argument("--eps-decay-steps", dest="eps_decay_steps", type=int, default=1000000,
                     help="The number of states eps anneal.")
    cmd.add_argument("--discount", dest="discount", type=float, default=0.99, help="The discount factor.")
    cmd.add_argument("--memory-size", dest="memory_size", type=int, default=1000000, help="The size of memory")
    cmd.add_argument("--batch-size", dest="batch_size", type=int, default=32, help="The number of samples each batch")
    cmd.add_argument("--target-update-freq", dest="target_update_freq", type=int, default=10000,
                     help="The frequency of update target network.")
    cmd.add_argument("--replay-start-size", dest="replay_start_size", type=int, default=50000,
                     help="The size of of states before replay start.")
    opts = cmd.parse_args()

    train_dataset = read_dataset(opts.reference)
    logging.info("Loaded {0} training sentences.".format(len(train_dataset)))
    devel_dataset = read_dataset(opts.development)
    logging.info("Loaded {0} development sentences.".format(len(devel_dataset)))
    test_dataset = read_dataset(opts.test)
    logging.info("Loaded {0} test sentences.".format(len(test_dataset)))

    form_alphabet = get_alphabet(train_dataset, 'form')
    logging.info("# {0} forms in alphabet".format(len(form_alphabet)))
    pos_alphabet = get_alphabet(train_dataset, 'pos')
    logging.info("# {0} postags in alphabet".format(len(pos_alphabet)))
    deprel_alphabet = get_alphabet(train_dataset, 'deprel')
    logging.info("# {0} deprel in alphabet".format(len(deprel_alphabet)))

    parser = Parser(form_alphabet, pos_alphabet, deprel_alphabet)
    model = DeepQNetwork(form_size=len(form_alphabet), form_dim=100, pos_size=len(pos_alphabet), pos_dim=20,
                         deprel_size=len(deprel_alphabet), deprel_dim=20, hidden_dim=opts.hidden_size,
                         output_dim=parser.num_actions(), dropout=opts.dropout, l2=opts.lamb)
    indices, matrix = load_embedding(opts.embedding, form_alphabet, opts.embedding_size)

    session = tf.Session()
    session.run(tf.initialize_all_variables())
    initialize_word_embeddings(session, model.form_emb, indices, matrix)
    logging.info('Embedding is loaded.')

    memory = []
    # starting from a random policy
    np.random.shuffle(train_dataset)
    n = 0
    while len(memory) < opts.replay_start_size:
        data = train_dataset[n]
        n += 1
        if n == len(train_dataset):
            n = 0
        if not is_projective(data) or not is_tree(data):
            logging.info("{0} is not tree or not projective, skipped.")
            continue
        d = [parser.ROOT] + data
        s = State(d)
        valid_ids, valid_names, _ = get_valid_actions(parser, s)
        while not s.terminate():
            x = parser.parameterize_x(parser.extract_features(s))
            i = np.random.randint(len(valid_names))
            chosen_id, chosen_name = valid_ids[i], valid_names[i]
            r = s.scored_transit(chosen_name)
            next_x = parser.parameterize_x(parser.extract_features(s))
            valid_ids, valid_names, valid_mask = get_valid_actions(parser, s)
            memory.append((x, chosen_id, r, next_x, valid_mask, s.terminate()))
    logging.info("Finish random initialization process, memory size {0}".format(len(memory)))

    # Learning DQN
    n, n_batch, iteration = 0, 0, 0
    best_uas = 0.

    eps = opts.eps_init
    eps_decay_rate = (opts.eps_init - opts.eps_final) / opts.eps_decay_steps
    logging.info('eps decay from {0} to {1} by {2} steps'.format(opts.eps_init, opts.eps_final, opts.eps_decay_steps))
    cost = 0.
    model.update_target(session)
    while iteration <= opts.max_iter:
        if n == 0:
            iteration += 1
            cost = 0
            logging.info("Start of iteration {0}, eps={1}, data shuffled.".format(iteration, eps))
            np.random.shuffle(train_dataset)
        data = train_dataset[n]

        if not is_projective(data) or not is_tree(data):
            logging.info("{0} is not tree or not projective, skipped.")
            continue

        d = [parser.ROOT] + data
        s = State(d)
        while not s.terminate():
            # eps-greedy, rollout policy
            p = np.random.rand()
            x = parser.parameterize_x(parser.extract_features(s))
            if p > eps:
                prediction = model.policy(session, [x])[0]
                best_score, best_id, best_name = find_best(parser, s, prediction)
                chosen_name, chosen_id = best_name, best_id
            else:
                valid_ids, valid_names, _ = get_valid_actions(parser, s)
                i = np.random.randint(len(valid_names))
                chosen_name, chosen_id = valid_names[i], valid_ids[i]
            r = s.scored_transit(chosen_name)
            next_x = parser.parameterize_x(parser.extract_features(s))
            valid_ids, valid_names, valid_mask = get_valid_actions(parser, s)
            memory.append((x, chosen_id, r, next_x, valid_mask, s.terminate()))
            if len(memory) > opts.memory_size:
                memory = memory[-opts.memory_size:]

            ids = np.random.choice(len(memory), opts.batch_size)
            xs = [memory[i][0] for i in ids]
            next_xs = [memory[i][3] for i in ids]
            actions = [memory[i][1] for i in ids]
            ys = np.array([memory[i][2] for i in ids], dtype=np.float32)
            next_ys = model.target_policy(session, next_xs)
            for row, i in enumerate(ids):
                next_ys[row, ~memory[i][4]] = np.NINF
            next_ys = np.amax(next_ys, axis=1) * opts.discount
            terminated_mask = np.array([memory[i][5] for i in ids], dtype=np.bool)
            next_ys[terminated_mask] = 0.
            ys += next_ys
            cost += model.train(session, xs, actions, ys)
            eps -= eps_decay_rate
            if eps < opts.eps_final:
                eps = opts.eps_final

            n_batch += 1
            if n_batch % opts.target_update_freq == 0:
                model.update_target(session)
                logging.info("target network is synchronized at {0}.".format(n_batch))

        # MOVE to the next instance.
        n += 1
        if (opts.evaluate_stops > 0 and n % opts.evaluate_stops == 0) or n == len(train_dataset):
            uas = evaluate(devel_dataset, session, parser, model)
            if n == len(train_dataset):
                logging.info('Iteration {0} done, eps={1}, Cost={2}, UAS={3}'.format(iteration, eps, cost, uas))
                n = 0
            else:
                logging.info('At {0}, eps={1} UAS={2}'.format(n_batch, eps, uas))
            if uas > best_uas:
                best_uas = uas
                uas = evaluate(test_dataset, session, parser, model)
                logging.info('New best achieved: {0}, test: {1}'.format(best_uas, uas))
    logging.info('Finish training, best uas is {0}'.format(best_uas))

if __name__ == "__main__":
    main()
