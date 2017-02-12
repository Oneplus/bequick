#!/usr/bin/env python
from __future__ import division
import numpy as np
try:
    from .instance_builder import InstanceBuilder as IB
except (ValueError, SystemError) as e:
    from instance_builder import InstanceBuilder as IB


class State(object):
    HED, DEPREL, L0, L1, R0, R1 = 0, 1, 2, 3, 4, 5
    ILL = 65535

    def __init__(self, data):
        self.n = len(data)
        self.data = data
        self.stack = [0]    # SHIFT in the pseudo ROOT
        self.buffer = range(1, len(data))
        self.result = np.full((self.n, 6), self.ILL, dtype=np.int32)

    def __str__(self):
        stack_str = str(self.stack)
        buffer_str = "[{0} .. {1}]".format(self.buffer[0], self.buffer[-1]) if len(self.buffer) > 0 else "[]"
        result_str = ", ".join(["{0}: {1}".format(i, res) for i, res in enumerate(self.result) if res[0] < self.ILL])
        return "stack: {0}\nbuffer: {1}\nresult: {2}".format(stack_str, buffer_str, result_str)

    def max_steps(self):
        return 2 * self.n - 2

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
        _mod[self.HED] = hed
        _mod[self.DEPREL] = deprel
        if _hed[self.L0] == self.ILL:
            _hed[self.L0] = mod
        elif mod < _hed[self.L0]:
            _hed[self.L1] = _hed[self.L0]
            _hed[self.L0] = mod
        elif mod < _hed[self.L1]:
            _hed[self.L1] = mod

    def right(self, deprel):
        mod = self.stack[-1]
        hed = self.stack[-2]
        self.stack = self.stack[:-2]
        self.stack.append(hed)
        _mod = self.result[mod]
        _hed = self.result[hed]
        _mod[self.HED] = hed
        _mod[self.DEPREL] = deprel
        if _hed[self.R0] == self.ILL:
            _hed[self.R0] = mod
        elif mod > _hed[self.R0]:
            _hed[self.R1] = _hed[self.R0]
            _hed[self.R0] = mod
        elif _hed[self.R1] == self.ILL or mod > _hed[self.R1]:
            _hed[self.R1] = mod

    def terminate(self):
        return len(self.stack) == 1 and len(self.buffer) == 0


class TransitionSystem(object):
    def __init__(self, deprel_alpha):
        self.deprel_alpha = deprel_alpha
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
        self.n_actions = len(self.deprel_alpha) * 2 - 3

    @classmethod
    def _is_int_shift(cls, a):
        return a == 0

    @classmethod
    def _is_str_shift(cls, a):
        return a == 'SH'

    @classmethod
    def _is_int_left(cls, a):
        return a > 0 and a % 2 == 1

    @classmethod
    def _is_str_left(cls, a):
        return a.startswith('LA-')

    @classmethod
    def _is_int_right(cls, a):
        return a > 0 and a % 2 == 0

    @classmethod
    def _is_str_right(cls, a):
        return a.startswith('RA-')

    @classmethod
    def _parse_label(cls, a, offset):
        return (a - 1) // 2 + 2 if offset else (a - 1) // 2

    @classmethod
    def _make_int_shift(cls, a=None):
        return 0

    @classmethod
    def _make_str_shift(cls, a=None):
        return 'SH'

    @classmethod
    def _make_int_left(cls, d):
        return 1 + 2 * d

    def _make_str_left(self, a):
        """

        :param a: int
        :return:
        """
        return 'LA-{0}'.format(self.deprel_cache[a])

    @classmethod
    def _make_int_right(cls, d):
        return 2 + 2 * d

    def _make_str_right(self, a):
        """

        :param a: int
        :return:
        """
        return 'RA-{0}'.format(self.deprel_cache[a])

    @classmethod
    def transit(cls, state, action):
        """

        :param state: State
        :param action: int
        :return:
        """
        if cls._is_int_left(action):
            state.left(cls._parse_label(action, True))
        elif cls._is_int_right(action):
            state.right(cls._parse_label(action, True))
        else:
            state.shift()

    @classmethod
    def scored_transit(cls, state, action):
        if cls._is_int_left(action):
            deprel = cls._parse_label(action, True)
            hed = state.stack[-1]
            mod = state.stack[-2]
            item = state.data[mod]
            score = (1. if item[IB.HED] == hed and item[IB.DEPREL] == deprel else -1.)
            state.left(deprel)
        elif cls._is_int_right(action):
            deprel = cls._parse_label(action, True)
            hed = state.stack[-2]
            mod = state.stack[-1]
            item = state.data[mod]
            score = (1. if item[IB.HED] == hed and item[IB.DEPREL] == deprel else -1.)
            state.right(deprel)
        else:
            state.shift()
            score = 0.
        return score

    def valid(self, state, action):
        """

        :param state: State
        :param action: int, action index
        :return:
        """
        if self._is_int_left(action):
            deprel = self._parse_label(action, True)
            is_root = self.deprel_cache[deprel] in ('root', 'ROOT', 'HED')
            if len(state.stack) < 2:
                return False
            if state.stack[-2] == 0:  # root not reduced
                return False
            if is_root:  # left root not allowed
                return False
        elif self._is_int_right(action):
            deprel = self._parse_label(action, True)
            is_root = self.deprel_cache[deprel] in ('root', 'ROOT', 'HED')
            if len(state.stack) < 2:
                return False
            if state.stack[-2] == 0:
                if not is_root or len(state.buffer) > 0:
                    return False
            elif is_root:
                return False
        else:
            if len(state.buffer) < 1:
                return False
        return True

    @classmethod
    def oracle_action(cls, s, data):
        """

        :param s: State
        :param data: list
        :return int:
        """
        top0 = s.stack[-1] if len(s.stack) > 0 else -1
        top1 = s.stack[-2] if len(s.stack) > 1 else -2
        all_descendants_reduced = True
        if top0 >= 0:
            for d in data:
                if d[IB.HED] == top0 and \
                        (s.result[d[IB.ID]][State.HED] == State.ILL or s.result[d[IB.ID]][State.HED] != top0):
                    all_descendants_reduced = False
                    break

        if top1 >= 0 and data[top1][IB.HED] == top0:
            return cls._make_int_left(data[top1][IB.DEPREL] - 2)
        elif top1 >= 0 and data[top0][IB.HED] == top1 and all_descendants_reduced:
            return cls._make_int_right(data[top0][IB.DEPREL] - 2)
        elif len(s.buffer) > 0:
            return cls._make_int_shift()

    def num_actions(self):
        return self.n_actions

    def get_action(self, a):
        if isinstance(a, str):
            return self._get_int_action(a)
        elif isinstance(a, int) or isinstance(a, long):
            return self._get_string_action(a)
        else:
            raise AttributeError("a: " + str(type(a)) + " is not support")

    def _get_string_action(self, a):
        return self.actions[a]

    def _get_int_action(self, a):
        return self.actions_cache[a]

    def get_actions(self):
        return self.actions


class Parser(object):
    S0, S1, S2, N0, N1, N2 = 0, 1, 2, 3, 4, 5
    S0L0, S0L1, S0R0, S0R1, S1L0, S1L1, S1R0, S1R1, S0LL, S0RR, S1LL, S1RR = 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
    _1ST_ORDER = [S0, S1, S2, N0, N1, N2]
    _2ND_ORDER = [S0L0, S0L1, S0R0, S0R1, S1L0, S1L1, S1R0, S1R1, S0LL, S0RR, S1LL, S1RR]

    FORM_NAMES = _1ST_ORDER + _2ND_ORDER
    POS_NAMES = _1ST_ORDER + _2ND_ORDER
    DEPREL_NAMES = _2ND_ORDER

    def __init__(self, system):
        """

        :param system: TransitionSystem
        """
        self.system = system

    def generate_training_instance(self, data):
        """

        :param data: list
        :return: tuple
        """
        s = State(data)
        n = s.max_steps()
        form = np.zeros((n, len(self.FORM_NAMES)), dtype=np.int32)
        pos = np.zeros((n, len(self.POS_NAMES)), dtype=np.int32)
        deprel = np.zeros((n, len(self.DEPREL_NAMES)), dtype=np.int32)
        oracle_actions = np.zeros(n, dtype=np.int32)
        step = 0
        while not s.terminate():
            action = oracle_actions[step] = TransitionSystem.oracle_action(s, data)
            form[step], pos[step], deprel[step] = self.parameterize_x(s)
            TransitionSystem.transit(s, action)
            step += 1
        return (form, pos, deprel), oracle_actions

    def _extract_features(self, s):
        """

        :param s: State
        :return ctx: array-like
        """
        ctx = np.full((18, ), State.ILL, dtype=np.int32)
        if len(s.stack) > 0:
            S0 = ctx[self.S0] = s.stack[-1]
            if s.result[S0][State.L0] != State.ILL:
                S0L0 = ctx[self.S0L0] = s.result[S0][State.L0]
                if s.result[S0L0][State.L0] != State.ILL:
                    ctx[self.S0LL] = s.result[S0L0][State.L0]
            if s.result[S0][State.L1] != State.ILL:
                ctx[self.S0L1] = s.result[S0][State.L1]
            if s.result[S0][State.R0] != State.ILL:
                S0R0 = ctx[self.S0R0] = s.result[S0][State.R0]
                if s.result[S0R0][State.R0] != State.ILL:
                    ctx[self.S0RR] = s.result[S0R0][State.R0]
            if s.result[S0][State.R1] != State.ILL:
                ctx[self.S0R1] = s.result[S0][State.R1]
        if len(s.stack) > 1:
            S1 = ctx[self.S1] = s.stack[-2]
            if s.result[S1][State.L0] != State.ILL:
                S1L0 = ctx[self.S1L0] = s.result[S1][State.L0]
                if s.result[S1L0][State.L0] != State.ILL:
                    ctx[self.S1LL] = s.result[S1L0][State.L0]
            if s.result[S1][State.L1] != State.ILL:
                ctx[self.S1L1] = s.result[S1][State.L1]
            if s.result[S1][State.R0] != State.ILL:
                S1R0 = ctx[self.S1R0] = s.result[S1][State.R0]
                if s.result[S1R0][State.R0] != State.ILL:
                    ctx[self.S1RR] = s.result[S1R0][State.R0]
            if s.result[S1][State.R1] != State.ILL:
                ctx[self.S1R1] = s.result[S1][State.R1]
        if len(s.stack) > 2:
            ctx[self.S2] = s.stack[-3]
        if len(s.buffer) > 0:
            ctx[self.N0] = s.buffer[0]
        if len(s.buffer) > 1:
            ctx[self.N1] = s.buffer[1]
        if len(s.buffer) > 2:
            ctx[self.N2] = s.buffer[2]
        return ctx

    def parameterize_x(self, s):
        """

        :param s: State
        :return: tuple
        """
        ctx, data, result = self._extract_features(s), s.data, s.result
        form = np.zeros((1, len(self.FORM_NAMES)), dtype=np.int32)
        tag = np.zeros((1, len(self.POS_NAMES)), dtype=np.int32)
        deprel = np.zeros((1, len(self.DEPREL_NAMES)), dtype=np.int32)
        for i, name in enumerate(self.FORM_NAMES):
            if ctx[name] != State.ILL:
                form[0, i] = data[ctx[name]][IB.FROM]
        for i, name in enumerate(self.POS_NAMES):
            if ctx[name] != State.ILL:
                tag[0, i] = data[ctx[name]][IB.POS]
        for i, name in enumerate(self.DEPREL_NAMES):
            if ctx[name] != State.ILL:
                deprel[0, i] = result[ctx[name]][State.DEPREL]
        return form, tag, deprel
