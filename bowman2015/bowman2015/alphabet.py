#!/usr/bin/env python


class Alphabet(object):
    """
    Convert the raw form of a string into its index. If configured,
    0 is placeholder of BAD0, 1 is placeholder of UNK.
    """
    def __init__(self, use_default_initialization=False):
        self.have_default_initialization = use_default_initialization

        if use_default_initialization:
            self.id2str = {0: "__BAD0__", 1: "__UNK__"}
            self.str2id = {"__BAD0__": 0, "__UNK__": 1}
        else:
            self.id2str = {}
            self.str2id = {}

    def insert(self, name):
        """

        :param name:
        :return:
        """
        new_id = len(self.id2str)
        self.id2str[new_id] = name
        self.str2id[name] = new_id

    def get(self, name):
        """

        :param name:
        :return:
        """
        if isinstance(name, str):
            if not name in self.str2id:
                raise NameError("name %s not found in alphabet." % name)
            return self.str2id.get(name)
        elif isinstance(name, int):
            if not name in self.id2str:
                raise NameError("index %d not found in alphabet." % name)
            return self.id2str.get(name)
        else:
            raise TypeError("Unsupported type in alphabet!")