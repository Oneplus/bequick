#!/usr/bin/env python
import platform
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
    from .tb_parser import TransitionSystem, Parser, State
    from .model import DeepQNetwork, initialize_word_embeddings
    from .tree_utils import is_projective_raw, is_tree_raw
    from .evaluate import evaluate
    from .instance_builder import InstanceBuilder
except (ValueError, SystemError) as e:
    from tb_parser import TransitionSystem, Parser, State
    from model import DeepQNetwork, initialize_word_embeddings
    from tree_utils import is_projective_raw, is_tree_raw
    from evaluate import evaluate
    from instance_builder import InstanceBuilder

np.random.seed(1234)
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
LOG = logging.getLogger('chen2014-rl')


class Memory(object):
    def __init__(self, n_actions, memory_size, batch_size):
        self.memory_size = memory_size
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.memory_volume = 0
        self.current_id = 0
        self.s_t_form = np.zeros((memory_size, len(Parser.FORM_NAMES)), dtype=np.int32)
        self.s_t_pos = np.zeros((memory_size, len(Parser.POS_NAMES)), dtype=np.int32)
        self.s_t_deprel = np.zeros((memory_size, len(Parser.DEPREL_NAMES)), dtype=np.int32)
        self.a_t = np.zeros(memory_size, dtype=np.int32)
        self.r_t = np.zeros(memory_size, dtype=np.float32)
        self.s_t_plus_1_form = np.zeros((memory_size, len(Parser.FORM_NAMES)), dtype=np.int32)
        self.s_t_plus_1_pos = np.zeros((memory_size, len(Parser.POS_NAMES)), dtype=np.int32)
        self.s_t_plus_1_deprel = np.zeros((memory_size, len(Parser.DEPREL_NAMES)), dtype=np.int32)
        self.s_t_plus_1_valid_mask = np.zeros((memory_size, n_actions), dtype=np.bool)
        self.terminate = np.zeros(memory_size, dtype=np.bool)

    def add(self, s_t_form, s_t_pos, s_t_deprel, a_t, r_t, s_t_plus_1_form, s_t_plus_1_pos, s_t_plus_1_deprel,
            s_t_valid_mask, terminate):
        self.s_t_form[self.current_id] = s_t_form
        self.s_t_pos[self.current_id] = s_t_pos
        self.s_t_deprel[self.current_id] = s_t_deprel
        self.a_t[self.current_id] = a_t
        self.r_t[self.current_id] = r_t
        self.s_t_plus_1_form[self.current_id] = s_t_plus_1_form
        self.s_t_plus_1_pos[self.current_id] = s_t_plus_1_pos
        self.s_t_plus_1_deprel[self.current_id] = s_t_plus_1_deprel
        self.s_t_plus_1_valid_mask[self.current_id] = s_t_valid_mask
        self.terminate[self.current_id] = terminate
        self.current_id = (self.current_id + 1) % self.memory_size
        self.memory_volume += 1
        if self.memory_volume >= self.memory_size:
            self.memory_volume = self.memory_size

    def volume(self):
        return self.memory_volume

    def sample(self):
        ids = np.random.choice(self.volume(), self.batch_size)
        return (self.s_t_form[ids], self.s_t_pos[ids], self.s_t_deprel[ids],
                self.a_t[ids], self.r_t[ids],
                self.s_t_plus_1_form[ids], self.s_t_plus_1_pos[ids], self.s_t_plus_1_deprel[ids],
                self.s_t_plus_1_valid_mask[ids], self.terminate[ids])


def find_best(parser, state, scores):
    best_score, best_i, best_name = None, None, None
    for i, score in enumerate(scores):
        name = parser.get_action(i)
        if state.valid(name) and (best_score is None or score > best_score):
            best_score, best_i, best_name = score, i, name
    return best_score, best_i, best_name


def get_valid_actions(parser, state):
    """

    :param parser: Parser
    :param state: State
    :return: tuple
    """
    n_actions = parser.system.num_actions()
    aids = []
    mask = np.zeros(n_actions, dtype=np.bool)
    for aid in range(n_actions):
        if parser.system.valid(state, aid):
            aids.append(aid)
            mask[aid] = True
    return aids, mask


def main():
    cmd = argparse.ArgumentParser("Learning component for chen and manning (2014)'s parser")
    cmd.add_argument("--model", help="The path to the model.")
    cmd.add_argument("--embedding", help="The path to the embedding file.")
    cmd.add_argument("--reference", help="The path to the reference file.")
    cmd.add_argument("--development", help="The path to the development file.")
    cmd.add_argument("--test", help="The path to the test file.")
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
    cmd.add_argument("--language", dest="lang", default="en", help="the language")
    opts = cmd.parse_args()

    raw_train = read_conllx_dataset(opts.reference)
    raw_devel = read_conllx_dataset(opts.development)
    raw_test = read_conllx_dataset(opts.test)
    LOG.info("Dataset stats: #train={0}, #devel={1}, #test={2}.".format(len(raw_train), len(raw_devel), len(raw_test)))
    raw_train = [data for data in raw_train if is_tree_raw(data) and is_projective_raw(data)]
    LOG.info("{0} training sentences after filtering non-tree and non-projective.".format(len(raw_train)))

    form_alphabet = get_alphabet(raw_train, 'form')
    pos_alphabet = get_alphabet(raw_train, 'pos')
    deprel_alphabet = get_alphabet(raw_train, 'deprel')
    form_alphabet['_ROOT_'] = len(form_alphabet)
    pos_alphabet['_ROOT_'] = len(pos_alphabet)
    LOG.info("Alphabet stats: #form={0} (w/ nil & unk & root), #pos={1} (w/ nil & unk & root),"
             " #deprel={2} (w/ nil & unk)".format(len(form_alphabet), len(pos_alphabet), len(deprel_alphabet)))

    instance_builder = InstanceBuilder(form_alphabet, pos_alphabet, deprel_alphabet)
    train_dataset = instance_builder.conllx_to_instances(raw_train, add_pseudo_root=True)
    devel_dataset = instance_builder.conllx_to_instances(raw_devel, add_pseudo_root=True)
    test_dataset = instance_builder.conllx_to_instances(raw_test, add_pseudo_root=True)
    devel_feats = [[None] + [token['feat'] for token in data] for data in raw_devel]
    test_feats = [[None] + [token['feat'] for token in data] for data in raw_test]
    LOG.info("Dataset converted from string to index.")
    LOG.info("{0} training sentences after filtering non-tree and non-projective.".format(len(train_dataset)))

    system = TransitionSystem(deprel_alphabet)
    parser = Parser(system)
    model = DeepQNetwork(form_size=len(form_alphabet), form_dim=100, pos_size=len(pos_alphabet), pos_dim=20,
                         deprel_size=len(deprel_alphabet), deprel_dim=20, hidden_dim=opts.hidden_size,
                         output_dim=system.num_actions(), dropout=opts.dropout, l2=opts.lamb)
    indices, matrix = load_embedding(opts.embedding, form_alphabet, opts.embedding_size)

    if platform.system() == 'Windows':
        session = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    else:
        session = tf.Session()
    session.run(tf.global_variables_initializer())
    initialize_word_embeddings(session, model.form_emb, indices, matrix)
    logging.info('Embedding is loaded.')

    memory = Memory(system.num_actions(), opts.memory_size, opts.batch_size)
    # starting from a random policy
    np.random.shuffle(train_dataset)
    n = 0
    while memory.volume() < opts.replay_start_size:
        data = train_dataset[n]
        n += 1
        if n == len(train_dataset):
            n = 0
        s = State(data)
        valid_ids, _ = get_valid_actions(parser, s)
        while not s.terminate():
            x = parser.parameterize_x(s)
            chosen_id = np.random.choice(valid_ids)
            r = system.scored_transit(s, chosen_id)
            next_x = parser.parameterize_x(s)
            valid_ids, valid_mask = get_valid_actions(parser, s)
            memory.add(x[0], x[1], x[2], chosen_id, r, next_x[0], next_x[1], next_x[2], valid_mask, s.terminate())
    LOG.info("Finish random initialization process, memory size {0}".format(memory.volume()))

    # Learning DQN
    n, n_batch, iteration = 0, 0, 0
    best_uas, test_uas, test_las = 0., 0., 0.

    eps = opts.eps_init
    eps_decay_rate = (opts.eps_init - opts.eps_final) / opts.eps_decay_steps
    LOG.info('eps decay from {0} to {1} by {2} steps'.format(opts.eps_init, opts.eps_final, opts.eps_decay_steps))
    cost = 0.
    while iteration <= opts.max_iter:
        if n == 0:
            cost = 0
            logging.info("Start of iteration {0}, eps={1}, data shuffled.".format(iteration, eps))
            np.random.shuffle(train_dataset)
        data = train_dataset[n]

        s = State(data)
        valid_ids, valid_mask = get_valid_actions(parser, s)
        while not s.terminate():
            # eps-greedy, rollout policy
            p = np.random.rand()
            x = parser.parameterize_x(s)
            if p > eps:
                prediction = model.target_policy(session, x)[0]
                prediction[~valid_mask] = np.NINF
                chosen_id = np.argmax(prediction).item()
            else:
                chosen_id = np.random.choice(valid_ids)
            r = system.scored_transit(s, chosen_id)
            next_x = parser.parameterize_x(s)
            valid_ids, valid_mask = get_valid_actions(parser, s)
            memory.add(x[0], x[1], x[2], chosen_id, r, next_x[0], next_x[1], next_x[2], valid_mask, s.terminate())

            payload = memory.sample()
            xs = payload[0], payload[1], payload[2]
            actions = payload[3]
            ys = payload[4].copy()
            next_xs = payload[5], payload[6], payload[7]
            next_valid_mask = payload[8]
            terminated = payload[9]

            next_ys = model.target_policy(session, next_xs)
            next_ys[~next_valid_mask] = -1e30
            ys += (1 - terminated) * np.amax(next_ys, axis=1) * opts.discount
            cost += model.train(session, xs, actions, ys)
            eps -= eps_decay_rate
            if eps < opts.eps_final:
                eps = opts.eps_final

            n_batch += 1
            if n_batch % opts.target_update_freq == 0:
                model.update_target(session)
                LOG.info("target network is synchronized at {0}.".format(n_batch))

        # MOVE to the next sentence.
        n += 1
        if (opts.evaluate_stops > 0 and n % opts.evaluate_stops == 0) or n == len(train_dataset):
            uas, las = evaluate(devel_dataset, session, parser, model, devel_feats, True, opts.lang)
            if n == len(train_dataset):
                LOG.info('Iteration {0} done, eps={1}, Cost={2}, UAS={3}, LAS={4}'.format(iteration, eps, cost,
                                                                                          uas, las))
                iteration += 1
                n = 0
            else:
                LOG.info('At {0}, eps={1} UAS={2}'.format(n_batch, eps, uas))
            if uas > best_uas:
                best_uas = uas
                test_uas, test_las = evaluate(test_dataset, session, parser, model, test_feats, True, opts.lang)
                LOG.info('New best achieved: {0}, test: UAS={1}, LAS={2}'.format(best_uas, test_uas, test_las))
    LOG.info('Finish training, best devel UAS={0}, test UAS={1}, LAS={2}'.format(best_uas, test_uas, test_las))

if __name__ == "__main__":
    main()
