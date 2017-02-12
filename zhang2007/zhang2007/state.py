#!/usr/bin/env python


class State(object):
    """
    The state object
    - score: The score to current state
    - index: The transition works between ith char and (i+1)th char
    - link:
    - action: current state = action(previous state)
    - prev: The index of previous word
    - curr: The index of current word
    """
    def __init__(self, score, index, state, action):
        self.score = score
        self.index = index
        self.link = state
        self.action = action

        if action == 'j':
            self.prev = state.prev
            self.curr = state.curr
        elif action == 's':
            self.prev = state.curr
            self.curr = self
        else:
            raise AttributeError("action name %s cannot be understood, expecting: j or s" % action)

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
