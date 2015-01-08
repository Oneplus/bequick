#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Yijia Liu"
__email__ = "oneplus.lau@gmail.com"

from heapq import heappush, heappop, heappushpop
from optparse import OptionParser

__boc__ = "__boc__"
__eoc__ = "__eoc__"
__bot__ = "__bot__"
__eot__ = "__eot__"
__bow__ = "__bow__"

class State(object):
    '''
    The state object
    '''
    def __init__(self, score, index, state, action):
        self.score = score          # The score to current state
        self.index = index          # The transition works between ith char and (i+1)th char
        self.link = state
        self.action = action

        if action == 'j':
            self.prev = state.prev
            self.curr = state.curr  # The previous begining of word link
        elif action == 's':
            self.prev = state.curr
            self.curr = self
        else:
            self.prev = None
            self.curr = self

    def __str__(self):
        ret = "ref: " + str(id(self))
        ret += " , index: " + str(self.index)
        ret += " , score:" + str(self.score)
        ret += " , prev:" + str(id(self.prev))
        ret += " , curr: " + str(id(self.curr))
        return str(ret)

    def __repr__(self):
        return self.__str__()

    def __cmp__(self, other):
        return cmp(self.score, other.score)


def chartype(ch):
    return 'other'


def get_gold_actions(words):
    '''
    Get gold transition actions from a list of words

    Parameters
    ----------
    words : The list of words

    Returns
    -------
    '''
    ret = []
    for word in words:
        chars = word.decode("utf-8")
        for idx, ch in enumerate(chars):
            if idx == len(chars) - 1:
                ret.append('s')
            else:
                ret.append('j')
    return ret

# jsjssjsjsjs
# print get_gold_actions(['浦东', '开发', '与', '法制', '建设', '同步'])

def number_of_characters(words):
    '''
    Get number of characters in a list of words
    '''
    ret = 0
    for word in words:
        ret += len(word.decode("utf-8"))
    return ret


def convert_words_to_characters(words):
    chars = []
    for word in words:
        chars.extend([ch.encode("utf-8") for ch in word.decode("utf-8")])

    return chars


def kmax_heappush(array, state, k):
    '''
    Insert the state into a k-max array
    '''
    if len(array) < k:
        heappush(array, state)
        return True
    elif len(array) == k:
        if array[0].score < state.score:
            heappushpop(array, state)
            return True
        return False
    return False


def extract_join_features(chars, charts, i, state):
    '''
    Extract features for JOIN action
    
    Paramters
    ---------
    chars : list(str)
        The list of characters
    charts : list(str)
        The list of character types
    i : int
        The index
    state : State
        The state

    Returns
    -------
    ret : list(str)
        The list of feature strings
    '''
    L = len(chars)

    prev_ch = chars[i - 1] if i - 1 >= 0 else __boc__
    curr_ch = chars[i]
    next_ch = chars[i + 1] if i + 1 < L else __eoc__
    prev_cht = charts[i - 1] if i - 1 >= 0 else __bot__
    curr_cht = charts[i]
    next_cht = charts[i + 1] if i + 1 < L else __eot__

    ret = ["1=c[-1]=%s" % prev_ch,
            "2=c[0]=%s" % curr_ch,
            "3=c[1]=%s" % next_ch,
            "4=ct[-1]=%s" % prev_cht,
            "5=ct[0]=%s" % curr_cht,
            "6=ct[1]=%s" % next_cht,
            "7=c[-1]c[0]=%s%s" % (prev_ch, curr_ch),
            "8=c[0]c[1]=%s%s" % (curr_ch, next_ch),
            "9=ct[-1]ct[0]=%s%s" % (prev_cht, curr_cht),
            "10=ct[0]ct[1]=%s%s" % (curr_cht, next_cht),
            "11=c[-1]c[0]c[1]=%s%s%s" % (prev_ch, curr_ch, next_ch),
            "12=ct[-1]ct[0]ct[1]=%s%s%s" % (prev_cht, curr_cht, next_cht)]
    return ret


def extract_separate_features(chars, charts, i, state):
    '''
    Extract features for the SEPARATE actions

    Parameters
    ----------
    chars : list(str)
        The list of characters
    charts : list(str)
        The list of character types
    i : int
        The index
    state : State
        The source state

    Returns
    -------
    ret : list(str)
        The list of feature string
    '''
    L = len(chars)

    prev_ch = chars[i - 1] if i - 1 >= 0 else __boc__
    curr_ch = chars[i]
    next_ch = chars[i + 1] if i + 1 < L else __eoc__
    prev_cht = charts[i - 1] if i - 1 >= 0 else __bot__
    curr_cht = charts[i]
    next_cht = charts[i + 1] if i + 1 < L else __eot__
    curr_w = "".join(chars[state.curr.index: i + 1])
    curr_w_len = i + 1 - state.curr.index
    if state.prev is not None:
        prev_w = "".join(chars[state.prev.curr.index: state.curr.index])
        prev_w_len = state.curr.index - state.prev.curr.index

    ret = ["1=c[-1]=%s" % prev_ch,
            "2=c[0]=%s" % curr_ch,
            "3=c[1]=%s" % next_ch,
            "4=ct[-1]=%s" % prev_cht,
            "5=ct[0]=%s" % curr_cht,
            "6=ct[1]=%s" % next_cht,
            "7=c[-1]c[0]=%s%s" % (prev_ch, curr_ch),
            "8=c[0]c[1]=%s%s" % (curr_ch, next_ch),
            "9=ct[-1]ct[0]=%s%s" % (prev_cht, curr_cht),
            "10=ct[0]ct[1]=%s%s" % (curr_cht, next_cht),
            "11=c[-1]c[0]c[1]=%s%s%s" % (prev_ch, curr_ch, next_ch),
            "12=ct[-1]ct[0]ct[1]=%s%s%s" % (prev_cht, curr_cht, next_cht)]
    ret.append(("13=w[0]=%s" % curr_w))
    if curr_w_len == 1:
        ret.append("14=single-char")
    else:
        ret.append("15=first=%s=last=%s" % (chars[state.curr.index], chars[i]))
        ret.append("16=first=%s=len[0]=%d" % (chars[state.curr.index], curr_w_len))
        ret.append("17=last=%s=len[0]=%d" % (chars[i], curr_w_len))

    if state.prev is not None:
        ret.append("18=word[-1]=%s-word[0]=%s" % (prev_w, curr_w))
        ret.append("19=prevch=%s-word[0]=%s" % (chars[state.curr.index-1], curr_w))
        ret.append("20=word[-1]=%s-prevch=%s" % (prev_w, chars[state.curr.index-1]))
        ret.append("21=word[-1]=%s-len[0]=%d" % (prev_w, curr_w_len))
        ret.append("22=word[0]=%s-len[-1]=%d" % (curr_w, prev_w_len))
    return ret


def extract_features(action, chars, charts, i, state):
    if action == 's':
        return extract_separate_features(chars, charts, i, state)
    else:
        return extract_join_features(chars, charts, i, state)


def transition_score(action, chars, charts, i, state, params):
    '''
    Compute transition score according to the action

    Parameters
    ----------
    state : State
        The state
    action : str
        The action str

    Returns
    -------
    ret : float
        The transition score
    '''
    ret = 0.
    L = len(chars)
    if action == 'j':
        for feature in extract_join_features(chars, charts, i, state):
            ret += params.get(feature, 0.)
    elif action == 's':
        for feature in extract_separate_features(chars, charts, i, state):
            ret += params.get(feature, 0.)
    return ret


def flush_parameters(params, now):
    '''
    At the end of each iteration, flash the parameters

    Parameters
    ----------
    param : dict 
        The parameters
    now : int
        The current time
    '''
    for feature in params:
        w = params[feature]
        w[2] += (now - w[1]) * w[0]
        w[1] = now


def update_parameters(features, params, now, scale):
    '''
    '''
    for feature in features:
        if feature not in params:
            params[feature] = [0, 0, 0]

        w = params[feature]
        elasped = now - w[1]
        upd = scale
        cur_val = w[0]

        w[0] = cur_val + upd
        w[2] += elasped * cur_val + upd
        w[1] = now


def backtrace_and_get_state_list(state):
    ret = []
    while state is not None:
        ret.append(state)
        state = state.link
    ret.reverse()
    return ret


def beam_search(train, words, beam_size, params, reference_actions):
    '''
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
    reference_action: list(str)
        The reference actions

    Return
    ------
    (int, )
    '''
    # Initalize the beam matrix
    chars = convert_words_to_characters(words)
    charts = [chartype(ch) for ch in chars]
    L = len(chars)
    beam = [[] for _ in xrange(L)]

    beam[0].append(State(score = 0., index = 0, state = None, action = None))
    correct_state = beam[0][0]

    for i in xrange(L - 1):
        for state in beam[i]:
            gain = transition_score('j', chars, charts, i, state, params)
            added = kmax_heappush(beam[i+1],
                    State(score = state.score + gain,
                        index = i + 1,
                        state = state,
                        action = 'j'),
                    beam_size)

            gain = transition_score('s', chars, charts, i, state, params)
            added = kmax_heappush(beam[i+1],
                    State(score = state.score + gain,
                        index = i + 1,
                        state = state,
                        action = 's'),
                    beam_size)

        if train:
            in_beam = False
            for state in beam[i+1]:
                if state.link == correct_state and state.action == reference_actions[i]:
                    in_beam = True
                    correct_state = state
                    break

            if not in_beam:
                # early update
                best_predict_state = max(beam[i+1], key=lambda s: s.score)
                predict_state_path = backtrace_and_get_state(best_predict_state)

                beam[i+1].append(State(score = 0, index = i + 1, state = correct_state, action = reference_actions[i]))
                correct_state_path = backtrace_and_get_state_list(beam[i+1][-1])

                assert(len(predict_state_path) == len(correct_state_path))

                update_start_position = -1
                for i in xrange(len(predict_state_path) - 1):
                    if predict_state_path[i] == correct_state_path[i] and predict_state_path[i+1].action != correct_state_path[i+1].action:
                        update_start_position = i
                        break

                for i in xrange(update_start_position, len(predict_state_path) - 1):
                    correct_features = extract_features(correct_state_path[i+1].action, chars, charts, i, correct_state_path[i])
                    predict_features = extract_features(predict_state_path[i+1].action, chars, charts, i, predict_state_path[i])

                    update_parameters(params, correct_features, 1.)
                    update_parameters(params, predict_features, -1.)


def learn(opts):
    try:
        fpi = open(opts.train, "r")
    except:
        print >> sys.stderr, "ERROR: Failed to open file."
        sys.exit(1)

    params = {}
    data = []
    for line in fpi:
        words = line.strip().split()
        data.append(words)

    for nr_iter in xrange(opts.iteration):
        print >> sys.stderr, "# Iteration %d" % (nr_iter + 1)

        for sentence in data:
            gold_actions = get_gold_actions(sentence)
            beam_search(sentence, opts.beam_size, params, gold_actions)


if __name__=="__main__":
    usage = "A Python implementation for Zhang and Clark (2007)"
    optparser = OptionParser(usage)
    optparser.add_option("-t", "--train", dest="train", help="use to specify training data")
    optparser.add_option("-d", "--dev", dest="dev", help="use to specify development data")
    optparser.add_option("-e", "--test", dest="test", help="use to specify test data")
    optparser.add_option("-i", "--iteration", dest="iteration", help="use to specify the maximum number of iteration")
    optparser.add_option("-b", "--beam-size", dest="beam_size", help="use to specify the size of the beam")
    opts, args = optparser.parse_args()

    if len(args) >= 1 and args[0] == "learn":
        learn(opts)
