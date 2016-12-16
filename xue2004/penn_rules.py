#!/usr/bin/env python
import re
from nltk.tree import ParentedTree

PUNC_REGEX= "[\\!\\?\\-\\_\\/\\&\\+\\.\\,\\`\\'\\(\\)\\[\\]\\{\\}\\\"\\:\\;]+"

def _is_punctuation(terminal_node):
    label = terminal_node.label()
    word = terminal_node[0]

    if label in ("IN", "TO", "RB", "AUX", "DT"):
        return False
    return re.match(PUNC_REGEX, word) is not None

def _get_last_token_textual(node):
    retval = node
    while not isinstance(retval[-1], str):
        retval = retval[-1]
    return retval

def is_coordinated(nonterminal_node):
    '''
    Return whether the input phrase is coordinated phrase.

    Parameters
    ----------
    node: nltk.tree.ParentedTree
        The node.

    Returns
    -------
    If the input phrase is a coordinated phrase, return True; otherwise
    return False.
    '''
    if node.label() == "UCP":
        return True
    if node._parent.size() < 2:
        return False

    start = 0
    for c in node:
        if _is_terminal_node(c) and _is_punctuation(c):
            start += 1
        else:
            break

    for i in xrange(start+ 1, len(node._parent)- 1):
        c = node._parent[i]
        if c.label() == "CONJP":
            return True
        if c.label() == "CC" and c[0].lowercase() in ["either", "neither"]:
            return True

    if node.label() not in ("NP", "NX", "NML", "NAC"):
        label = None
        saw_comma = False
        uniform = True
        saw_token = False

        n = 0
        for c in node:
            if _is_nonterminal_node(c):
                n += 1
                if label is None:
                    label = c.label()
                    if label in ("SINV", "SQ", "SBARQ"):
                        label = "S"
                    else:
                        label2 = c.label()
                        if label2 in ("SINV", "SQ", "SBARQ"):
                            label2 = "S"
                        if label != label2:
                            uniform = False
            elif _is_terminal_node(c):
                if not _is_punctuation(c) and c.label() not in ("RB", "UH", "IN", "CC"):
                    saw_token = True

        saw_comma = False
        if len(node._parent) > 1:
            for c in node._parent[1:]:
                if c.label() in (",", ":"):
                    saw_comma = True

        if not saw_comma:
            for c in node._parent:
                if _get_last_token_textual(c).label() in (",", ":"):
                    saw_comma = True

        if saw_comma and uniform and n > 1:
            if saw_token:
                return False
            else:
                return True
    else:
        saw_comma = False
        for c in node:
            if c.label() in (",", ":"):
                saw_comma = True
                break
        if not saw_comma:
            return False
        nNP = 0
        for c in node:
            if c.label() in ("NP", "NX", "NML", "NAC"):
                if len(c) == 1 and c[0].label() == "CD":
                    pass
                else:
                    nNP += 1
        if nNP > 2:
            return True

    return False


# Testing
import unittest

class PennRulesUnittest(unittest.TestCase):
    def setUp(self):
        token = "(S1 (S (NP (NNP Ms.) (NNP Haag)) (VP (VBZ plays) (NP (NNP Elianti))) (. .)))"
        self.tree = ParentedTree.fromstring(token)

    test_1 = lambda self: self.assertEqual("S1", self.tree.label())
    test_2 = lambda self: self.assertTrue(_is_terminal_node(self.tree[0][1][0])) # S1-[0]->S-[1]->VP-[0]->VBZ
    test_3 = lambda self: self.assertEqual(_get_last_token_textual(self.tree[0][1]), ParentedTree.fromstring("(NNP Elianti)"))

if __name__=="__main__":
    unittest.main()
