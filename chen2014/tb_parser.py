#!/usr/bin/env python
import copy


class State(object):
    def __init__(self, data):
        self.n = len(data)
        self.data = data
        self.stack = []
        self.buffer = range(len(data))
        self.result = [{} for _ in range(len(data))]

    def __str__(self):
        stack_str = str(self.stack)
        buffer_str = "[{0} .. {1}]".format(self.buffer[0], self.buffer[1]) if len(self.buffer) > 0 else "[]"
        result_str = ", ".join(["{0}: {1}".format(i, res) for i, res in enumerate(self.result) if len(res) > 0])
        return "stack: {0}\nbuffer: {1}\nresult: {2}".format(stack_str, buffer_str, result_str)

    def shift(self):
        self.stack.append(self.buffer[0])
        self.buffer = self.buffer[1:]

    def left(self, deprel):
        hed = self.stack[-1]
        mod = self.stack[-2]
        self.stack = self.stack[:-2]
        self.stack.append(hed)
        _mod = self.result[mod]
        _hed = self.result[hed]
        _mod['head'] = hed
        _mod['deprel'] = deprel
        if 'L0' not in _hed:
            _hed['L0'] = mod
        elif mod < _hed['L0']:
            _hed['L1'] = _hed['L0']
            _hed['L0'] = mod
        else:
            _hed['L1'] = mod

    def right(self, deprel):
        mod = self.stack[-1]
        hed = self.stack[-2]
        self.stack = self.stack[:-2]
        self.stack.append(hed)
        _mod = self.result[mod]
        _hed = self.result[hed]
        _mod['head'] = hed
        _mod['deprel'] = deprel
        if 'R0' not in _hed:
            _hed['R0'] = mod
        elif mod > _hed['R0']:
            _hed['R1'] = _hed['R0']
            _hed['R0'] = mod
        else:
            _hed['R1'] = mod

    def terminate(self):
        return len(self.stack) == 1 and len(self.buffer) == 0

    def oracle_action(self, data):
        top0 = self.stack[-1] if len(self.stack) > 0 else -1
        top1 = self.stack[-2] if len(self.stack) > 1 else -2
        all_descendants_reduced = True
        if top0 >= 0:
            for d in data:
                if d['head'] == top0 and ('head' not in self.result[d['id']] or self.result[d['id']]['head'] != top0):
                    all_descendants_reduced = False
                    break

        if top1 >= 0 and data[top1]['head'] == top0:
            return 'LA-{0}'.format(data[top1]['deprel'])
        elif top1 >= 0 and data[top0]['head'] == top1 and all_descendants_reduced:
            return 'RA-{0}'.format(data[top0]['deprel'])
        elif len(self.buffer) > 0:
            return 'SH'

    def transit(self, action):
        if action.startswith('LA'):
            self.left(action.split('-')[1])
        elif action.startswith('RA'):
            self.right(action.split('-')[1])
        else:
            self.shift()

    def scored_transit(self, action):
        if action.startswith('LA'):
            hed = self.stack[-1]
            mod = self.stack[-2]
            score = 1. if self.data[mod]['head'] == hed else -1.
            self.left(action.split('-')[1])
        elif action.startswith('RA'):
            hed = self.stack[-2]
            mod = self.stack[-1]
            self.right(action.split('-')[1])
            score = 1. if self.data[mod]['head'] == hed else -1.
        else:
            self.shift()
            score = 0.
        return score

    def valid(self, action):
        if action.startswith('LA'):
            if len(self.stack) < 2:
                return False
            if self.stack[-2] == 0:  # root not reduced
                return False
            if action.split('-')[1] in ('root', 'ROOT', 'HED'): # left root not allowed
                return False
        elif action.startswith('RA'):
            if len(self.stack) < 2:
                return False
            if self.stack[-2] == 0:
                if action.split('-')[1] not in ('root', 'ROOT', 'HED') or len(self.buffer) > 0:
                    return False
        else:
            if len(self.buffer) < 1:
                return False
        return True

    def copy(self):
        new_state = State(self.data)
        new_state.stack = copy.deepcopy(self.stack)
        new_state.buffer = copy.deepcopy(self.buffer)
        new_state.result = copy.deepcopy(self.result)
        return new_state


class Parser(object):
    ROOT = {'id': 0, 'form': '_ROOT_', 'pos': '_ROOT_', 'head': None, 'deprel': None}

    def __init__(self, form_alpha, pos_alpha, deprel_alpha):
        self.form_alpha = form_alpha
        self.pos_alpha = pos_alpha
        self.deprel_alpha = deprel_alpha
        self.pos_cache = {}
        for k, v in pos_alpha.items():
            self.pos_cache[v] = k
        self.deprel_cache = {}
        self.actions = ['SH']
        self.actions_cache = {'SH': 0}
        for k, v in deprel_alpha.items():
            self.deprel_cache[v] = k
            if v > 1:
                name = 'LA-{0}'.format(k)
                self.actions_cache[name] = len(self.actions)
                self.actions.append(name)
                name = 'RA-{0}'.format(k)
                self.actions_cache[name] = len(self.actions)
                self.actions.append(name)

    def generate_training_instance(self, data):
        d = [self.ROOT] + data
        s = State(d)
        instances, oracle_actions = [], []
        while not s.terminate():
            action = s.oracle_action(d)
            oracle_actions.append(action)
            instances.append(self.extract_features(s))
            s.transit(action)
        return self.parameterize_X(instances, s), self.parameterize_Y(oracle_actions)

    def extract_features(self, s):
        ctx = {
            'S0': (s.stack[-1] if len(s.stack) > 0 else None),
            'S1': (s.stack[-2] if len(s.stack) > 1 else None),
            'S2': (s.stack[-3] if len(s.stack) > 2 else None),
            'N0': (s.buffer[0] if len(s.buffer) > 0 else None),
            'N1': (s.buffer[1] if len(s.buffer) > 1 else None),
            'N2': (s.buffer[2] if len(s.buffer) > 2 else None),
        }
        S0 = ctx['S0']
        ctx['S0L0'] = None if S0 is None else (None if 'L0' not in s.result[S0] else s.result[S0]['L0'])
        ctx['S0L1'] = None if S0 is None else (None if 'L1' not in s.result[S0] else s.result[S0]['L1'])
        ctx['S0R0'] = None if S0 is None else (None if 'R0' not in s.result[S0] else s.result[S0]['R0'])
        ctx['S0R1'] = None if S0 is None else (None if 'R1' not in s.result[S0] else s.result[S0]['R1'])
        ctx['S0LL'] = None if not ctx['S0L0'] else (None if 'L0' not in s.result[ctx['S0L0']] else s.result[ctx['S0L0']]['L0'])
        ctx['S0RR'] = None if not ctx['S0R0'] else (None if 'R0' not in s.result[ctx['S0R0']] else s.result[ctx['S0R0']]['R0'])
        S1 = ctx['S1']
        ctx['S1L0'] = None if S1 is None else (None if 'L0' not in s.result[S1] else s.result[S1]['L0'])
        ctx['S1L1'] = None if S1 is None else (None if 'L1' not in s.result[S1] else s.result[S1]['L1'])
        ctx['S1R0'] = None if S1 is None else (None if 'R0' not in s.result[S1] else s.result[S1]['R0'])
        ctx['S1R1'] = None if S1 is None else (None if 'R1' not in s.result[S1] else s.result[S1]['R1'])
        ctx['S1LL'] = None if not ctx['S1L0'] else (None if 'L0' not in s.result[ctx['S1L0']] else s.result[ctx['S1L0']]['L0'])
        ctx['S1RR'] = None if not ctx['S1R0'] else (None if 'R0' not in s.result[ctx['S1R0']] else s.result[ctx['S1R0']]['R0'])
        return ctx

    def get_oracle_actions(self, data):
        ret = []
        d = [self.ROOT] + data
        s = State(d)
        while not s.terminate():
            action = s.oracle_action(d)
            ret.append(action)
            s.transit(action)
        return ret

    _1ST_ORDER = ['S0', 'S1', 'S2', 'N0', 'N1', 'N2']
    _2ND_ORDER = ['S0L0', 'S0L1', 'S0R0', 'S0R1', 'S1L0', 'S1L1', 'S1R0', 'S1R1', 'S0LL', 'S0RR', 'S1LL', 'S1RR']

    FORM_NAMES = _1ST_ORDER + _2ND_ORDER
    POS_NAMES = _1ST_ORDER + _2ND_ORDER
    DEPREL_NAMES = _2ND_ORDER

    def parameterize_X(self, instances, s):
        ret = []
        for ctx in instances:
            forms, postags, deprels = [], [], []
            for name in self.FORM_NAMES:
                forms.append(self.form_alpha.get(s.data[ctx[name]]['form'], 1) if ctx[name] else 0)
            for name in self.POS_NAMES:
                postags.append(self.pos_alpha.get(s.data[ctx[name]]['pos']) if ctx[name] else 0)
            for name in self.DEPREL_NAMES:
                deprels.append(self.deprel_alpha.get(s.result[ctx[name]]['deprel']) if ctx[name] else 0)
            ret.append((forms, postags, deprels))
        return ret

    def parameterize_Y(self, actions):
        return [self._get_int_action(action) for action in actions]

    def num_actions(self):
        return len(self.deprel_alpha) * 2 - 3  # counting from 2, None for 0, UNK for 1

    def get_action(self, a):
        if isinstance(a, str):
            return self._get_int_action(a)
        elif isinstance(a, int):
            return self._get_string_action(a)
        else:
            raise AttributeError("a: " + type(a) + " is not support")

    def _get_string_action(self, a):
        return self.actions[a]

    def _get_int_action(self, a):
        return self.actions_cache[a]

    def get_actions(self):
        return self.actions
