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

    def __len__(self):
        return len(self.str2id)

    def insert(self, name):
        """

        :param name: str, the
        :return:
        """
        if name not in self.str2id:
            new_id = len(self.id2str)
            self.id2str[new_id] = name
            self.str2id[name] = new_id
        return self.str2id[name]

    def get(self, name):
        """

        :param name: str,
        :return:
        """
        if isinstance(name, str):
            if name not in self.str2id:
                if not self.have_default_initialization:
                    raise NameError("name %s not found in alphabet." % name)
                else:
                    return self.str2id.get("__UNK__")
            return self.str2id.get(name)
        elif isinstance(name, int):
            if name not in self.id2str:
                if not self.have_default_initialization:
                    raise NameError("index %d not found in alphabet." % name)
                else:
                    return self.id2str.get(1)
            return self.id2str.get(name)
        else:
            raise TypeError("Unsupported type in alphabet!")

    def __contains__(self, item):
        """

        :param item:
        :return:
        """
        if isinstance(item, str):
            return item in self.str2id
        elif isinstance(item, int):
            return item in self.id2str
        else:
            raise TypeError("Unsupported type in alphabet!")
