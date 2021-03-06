#!/usr/bin/env python
from __future__ import print_function
from nltk.tree import ParentedTree
import unittest


class Constituent(ParentedTree):
    def __init__(self, node, children=None):
        self.head_word = None
        self.start = None
        self.end = None
        super(Constituent, self).__init__(node, children)

    def terminal(self):
        return isinstance(self[0], str)

    def nonterminal(self):
        return not isinstance(self[0], str)

    def build_index(self, start):
        if self.terminal():
            self.start = start
            self.end = start + 1
        else:
            self.start = start
            for kid in self:
                kid.build_index(start)
                start = kid.end
            self.end = self[-1].end


def path(x, y):
    """
    Parameter
    ---------
    x: Constituent
        The 
    y: Constituent
        The
    """
    def path_to_root(t):
        output = []
        while t.parent() is not None:
            output.append(t)
            t = t.parent()
        return output
    xpath = path_to_root(x)
    ypath = path_to_root(y)
    lca = None
    for x_, y_ in zip(xpath[::-1], ypath[::-1]):
        if id(x_) == id(y_):
            continue
        lca = x_.parent()
        break
    assert lca is not None, "lca should not be None."
    upstream = []
    p = x
    while p != lca:
        upstream.append(("U", p))
        p = p.parent()
    downstream = []
    p = y
    while p != lca:
        downstream.append(("D", p))
        p = p.parent()
    return upstream + [("P", lca)] + downstream[::-1]


def build_constituent_tree(words, brackets):
    assert(len(words) == len(brackets))
    # convert the input into the penn bracketed reprensentation.
    txt = ""
    for word, bracket in zip(words, brackets):
        pos, payload = bracket.strip().split()
        pos = pos.replace("(", "-LBR-")
        pos = pos.replace(")", "-RBR-")
        word = word.replace("(", "-LBR-")
        word = word.replace(")", "-RBR-")
        txt += payload.replace("*", ("(%s %s)" % (pos, word)))
    t = Constituent.fromstring(txt)
    t.build_index(0)
    return t


def generate_candidate_constituents(t, predicate):
    """
    Travel to the top of the tree from the predicate and collect all the
    siblings. Since the `nltk.tree.ParentedTree.left_sibling` and
    `nltk.tree.ParentedTree.right_sibling` only return the first left-right
    sibling, code is hacked according to: 
    http://www.nltk.org/_modules/nltk/tree.html
    """
    retval = []
    now = predicate
    while now.parent() is not None:
        parent_index = now.parent_index()
        for i in range(0, parent_index):
            left_sibling = now._parent[i]
            retval.append(left_sibling)
            phrase = left_sibling.label().split("-")[0]
            if phrase == "PP":
                for kid in left_sibling:
                    retval.append(kid)
        for i in range(parent_index + 1, len(now._parent)):
            right_sibling = now._parent[i]
            retval.append(right_sibling)
            phrase = right_sibling.label().split("-")[0]
            if phrase == "PP":
                for kid in right_sibling:
                    retval.append(kid)
        now = now.parent()
    return retval


def locate_node(t, slc, extra):
    if t.start == slc[0] and t.end == slc[1] and extra(t):
        return t
    for kid in t:
        if kid.start <= slc[0] and slc[1] <= kid.end:
            return locate_node(kid, slc, extra)


class ConstituentUnittest(unittest.TestCase):
    def setUp(self):
        tkn = "(S1 (S (NP (NNP Ms.) (NNP Haag)) (VP (VBZ plays) (NP (NNP Elianti))) (. .)))"
        self.t = Constituent.fromstring(tkn)
        self.t.build_index(0)

    def test1(self): self.assertEqual("S1", self.t.label())

    def test2(self): self.assertTrue(self.t[0].nonterminal())

    def test3(self): self.assertTrue(self.t[0][0][0].terminal())

    def test4(self): self.assertEqual(0, self.t[0].start)

    def test5(self): self.assertEqual(5, self.t[0].end)

    def test6(self): self.assertEqual(2, self.t[0][1].start)

    def test7(self): self.assertEqual(4, self.t[0][1].end)

    def test8(self):
        for d, p in path(self.t[0][0][0], self.t[0][1][1]):
            print(d, p.pprint())

if __name__ == "__main__":
    unittest.main()
