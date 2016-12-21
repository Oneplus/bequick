#!/usr/bin/env python


class InstanceBuilder(object):
    ID, FROM, POS, HED, DEPREL = 0, 1, 2, 3, 4

    def __init__(self, form_alphabet, pos_alphabet, deprel_alphabet):
        self.form_alphabet = form_alphabet
        self.pos_alphabet = pos_alphabet
        self.deprel_alphabet = deprel_alphabet
        self.pseudo_root = (0, form_alphabet.get('_ROOT_'), pos_alphabet.get('_ROOT_'), None, None)

    def _conllx_to_instance(self, data, add_pseudo_root=False):
        ret = []
        if add_pseudo_root:
            ret.append(self.pseudo_root)
        for token in data:
            ret.append((token['id'],
                        self.form_alphabet.get(token['form'], 1),
                        self.pos_alphabet.get(token['pos']),
                        token['head'],
                        self.deprel_alphabet.get(token['deprel']))
                       )
        return ret

    def conllx_to_instances(self, dataset, add_pseudo_root=False):
        ret = []
        for data in dataset:
            ret.append(self._conllx_to_instance(data, add_pseudo_root))
        return ret
