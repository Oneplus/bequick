#!/usr/bin/env python


class Alphabet(object):
    def __init__(self):
        self.s2i = {}
        self.i2s = {}
        self.n = 0

    def add(self, name, idx = None):
        """

        :param name: str, the add string
        :param idx: int|None, the index, if idx is None use
        :return: none
        """
        assert isinstance(name, str)
        assert (isinstance(idx, int) or idx is None)
        new_idx = self.n if idx is None else idx
        if name not in self.s2i:
            self.s2i[name] = new_idx
            self.i2s[new_idx] = name
            self.n += 1

    def get(self, item):
        """
        Get
        :param item:
        :return:
        """
        if isinstance(item, str):
            return self.s2i.get(item, 0)
        elif isinstance(item, int):
            return self.i2s.get(item, None)
        else:
             raise AttributeError("Illegal type: %s. Expecting \"str\" or \"int\"" % type(item))

    def __contains__(self, item):
        """
        Find if the item is in the alphabet.
        :param item: str|int
        :return: bool, return True if the input item is in the alphabet, otherwise False.
        """
        if isinstance(item, str):
            return item in self.s2i
        elif isinstance(item, int):
            return item in self.i2s
        else:
            raise AttributeError("Illegal type: %s. Expecting \"str\" or \"int\"" % type(item))

