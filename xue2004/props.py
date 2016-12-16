#!/usr/bin/env python
from __future__ import print_function
import re
from xue2004.constituent import locate_node


def get_predicate_positions(prop):
    """
    Get the position of the predicate.

    Parameters
    ----------
    prop: str
        The string props matrix

    Return
    ------
    retval: list of tuple 2
        Each element indicate the starting and ending position of the predicate.
    """
    mat = [line.strip().split() for line in prop]
    nr_rows, nr_columns = len(mat), len(mat[0])
    for row in mat:
        assert(len(row) == nr_columns)

    retval = []
    for i in range(1, nr_columns):
        key = None
        start, end = None, None
        for j in xrange(0, nr_rows):
            token = mat[j][i]
            if token.startswith("(V*"):
                start = j
            if start is not None and token.endswith(")"):
                end = j + 1
                break
        assert start is not None and end is not None
        retval.append((start, end))
    return retval


def get_arguments_of_predicate(prop, predicate):
    """
    Get the arguments of the predicate.

    Parameters
    ----------
    prop: str
        The matrix for props
    predicate: Tree
        The node of predicate

    Return
    ------
    retval: list
    """
    mat = [line.strip().split() for line in prop]
    nr_rows, nr_columns = len(mat), len(mat[0])
    for row in mat:
        assert len(row) == nr_columns, "nr columns not equals"

    for i in range(1, nr_columns):
        start, end = None, None
        for j in range(0, nr_rows):
            token = mat[j][i]
            if token.startswith("(V*"):
                start = j
            if start is not None and token.endswith(")"):
                end = j + 1
                break
        assert start is not None and end is not None, "predicates not correctly located."
        if start == predicate.start and end == predicate.end:
            break

    status, start, end = None, -1, -1
    retval = []
    for j in range(0, nr_rows):
        token = mat[j][i]
        if token.startswith("("):
            m = re.search("\(([^\*]+)\*", token)
            assert(m is not None)
            status = m.group(1)
            start = j
        if token.endswith(")"):
            assert(status is not None)
            end = j + 1
            if status != "V":
                # assert status not in retval, "same argument type occurs multiple times."
                retval.append((status, start, end))
    return retval


def locate_predicate(t, slc):
    """
    Get the tree element of the predicate.

    Parameters
    ----------
    t:
        something
    slc: int or tuple
        The position of the predicate

    Return
    ------
    retval: nltk.tree.Constituent
        The tree node.
    """
    retval = None
    if isinstance(slc, tuple):
        assert t.start <= slc[0] and t.end >= slc[1], "{0}, {1}, {2}, {3}, {4}".format(t.start, t.end, slc[0], slc[1],
                                                                                       t.pprint())
        retval = locate_node(t, slc, lambda t: t.label() != "S")
        # If cannot allocate in the right range, return the verb
        if retval is None:
            for i in xrange(slc[0], slc[1]):
                retval = t[t.leaf_treeposition(i)[:-1]]
                if retval.label().startswith("V"):
                    break
    elif isinstance(slc, int):
        retval = t[t.leaf_treeposition(slc)[:-1]]
    return retval


if __name__ == "__main__":
    text = '''\
-                    (A0*      
-                       *)     
play                  (V*)     
-                    (A1*)     
-                       *      '''
    text = text.split("\n")
    ret = get_predicate_arguments(text)
    print(ret)
