#!/usr/bin/env python
import sys
import argparse
import logging
from zhang2007.state import State
from zhang2007.oracle import get_gold_actions
from zhang2007.util import chartype, number_of_characters, convert_words_to_characters, kmax_heappush
from zhang2007.model import flush_parameters, update_parameters
from zhang2007.extract import extract_features


def transition_score(train, action, chars, charts, i, state, params):
    """
    Compute transition score according to the action

    Parameters
    ----------
    train: bool
        If perform training.
    action : str
        The action str
    chars: list(str)
        The characters
    charts: list(str)
        The type of characters
    i: int
        The position
    state : State
        The state
    params: dict
        The parameters

    Returns
    -------
    ret : float
        The transition score
    """
    k = 0 if train else 1
    return sum(params.get(feature, [0., 0., 0])[k] for feature in extract_features(action, chars, charts, i, state))


def backtrace_and_get_state_list(state):
    ret = []
    while state is not None:
        ret.append(state)
        state = state.link
    ret.reverse()
    return ret


def beam_search(train, words, beam_size, params, timestamp, reference_actions):
    """
    Parameters
    ----------
    train: bool
        If run training process
    words: list(str)
        The words list
    beam_size: int
        The size of beam
    params: dict
        The parameters
    timestamp: int
        The timestamp of current instance
    reference_actions: list(str)
        The reference actions

    Return
    ------
    (int, )
    """
    # Initalize the beam matrix
    chars = convert_words_to_characters(words)
    charts = [chartype(ch) for ch in chars]
    L = len(chars)
    beam = [[] for _ in range(L)]

    beam[0].append(State(score=0., index=0, state=None, action=None))
    correct_state = beam[0][0]

    for i in range(L - 1):
        for state in beam[i]:
            for a in ['j', 's']:
                gain = transition_score(train, a, chars, charts, i, state, params)
                kmax_heappush(beam[i + 1],
                              State(score=state.score + gain, index=i + 1, state=state, action='j'),
                              beam_size)

        if train:
            in_beam = False
            for state in beam[i + 1]:
                if state.link == correct_state and state.action == reference_actions[i]:
                    in_beam = True
                    correct_state = state
                    break

            if not in_beam:
                # early update
                best_predict_state = max(beam[i + 1], key=lambda s: s.score)
                predict_state_path = backtrace_and_get_state_list(best_predict_state)

                beam[i + 1].append(State(score=0, index=i + 1, state=correct_state, action=reference_actions[i]))
                correct_state_path = backtrace_and_get_state_list(beam[i + 1][-1])

                assert (len(predict_state_path) == len(correct_state_path))

                update_start_position = -1
                for k in range(len(predict_state_path) - 1):
                    if predict_state_path[k] == correct_state_path[k] and predict_state_path[k + 1].action != \
                            correct_state_path[k + 1].action:
                        update_start_position = k
                        break

                for k in range(update_start_position, len(predict_state_path) - 1):
                    correct_features = extract_features(correct_state_path[k + 1].action, chars, charts, k,
                                                        correct_state_path[k])
                    predict_features = extract_features(predict_state_path[k + 1].action, chars, charts, k,
                                                        predict_state_path[k])

                    update_parameters(params, correct_features, timestamp, 1.)
                    update_parameters(params, predict_features, timestamp, -1.)
                return False, []
    best_predict_state = max(beam[L], key=lambda s: s.score)

    return True, backtrace_and_get_state_list(best_predict_state)


def learn():
    """
    Main learning function.
    """
    usage = "A Python implementation for Zhang and Clark (2007), learning components."
    opt_parser = argparse.ArgumentParser(usage)
    opt_parser.add_argument("-t", "--train", dest="train", help="use to specify training data")
    opt_parser.add_argument("-d", "--dev", dest="dev", help="use to specify development data")
    opt_parser.add_argument("-i", "--iteration", dest="iteration", help="use to specify the maximum number of iteration")
    opt_parser.add_argument("-b", "--beam-size", dest="beam_size", help="use to specify the size of the beam")
    opt_parser.add_argument("-m", "--model", dest="model", help="the path to the model")
    opts, args = opt_parser.parse_args()

    params = {}
    data = []
    for line in open(opts.train, "r"):
        words = line.strip().split()
        data.append(words)

    timestamp = 0
    for n_iter in range(opts.iteration):
        logging.info("start of iteration %d" % (n_iter + 1))
        n_error = 0
        for sentence in data:
            gold_actions = get_gold_actions(sentence)
            err, _ = beam_search(True, sentence, opts.beam_size, params, timestamp, gold_actions)
            if err:
                n_error += 1
            timestamp += 1
        flush_parameters(params, timestamp)
        logging.info("end of iteration %d" % (n_iter + 1))
        logging.info("number of errors in iteration %d" % (n_iter + 1))


def test():
    usage = "A Python implementation for Zhang and Clark (2007), learning components."
    opt_parser = argparse.ArgumentParser(usage)
    opt_parser.add_argument("-e", "--test", dest="test", help="use to specify test data")
    opt_parser.add_argument("-b", "--beam-size", dest="beam_size", help="use to specify the size of the beam")
    opt_parser.add_argument("-m", "--model", dest="model", help="the path to the model")
    opts, args = opt_parser.parse_args()


if __name__ == "__main__":
    if len(sys.argv) >= 1 and sys.argv[1] == "learn":
        learn()
    elif len(sys.argv) >= 1 and sys.argv[1] == "test":
        test()
    else:
        print >> sys.stderr, ("%s [learn|test]" % sys.argv[0])
