#!/usr/bin/env python
import re
from constituent import locate_node

def get_props_indices(prop):
    mat = [line.strip().split() for line in prop]
    nr_rows, nr_columns = len(mat), len(mat[0])
    for row in mat:
        assert(len(row) == nr_columns)

    retval = []
    for i in xrange(1, nr_columns):
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


def get_predicate_arguments(prop, predicate):
    mat = [line.strip().split() for line in prop]
    nr_rows, nr_columns = len(mat), len(mat[0])
    for row in mat:
        assert(len(row) == nr_columns)

    retval = {}
    for i in xrange(1, nr_columns):
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
        if start == predicate.start and end == predicate.end:
            break

    status, start, end = None, -1, -1
    for j in xrange(0, nr_rows):
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
                retval[status] = (start, end)
    return retval


def locate_predicate(t, slc):
    retval = None
    if isinstance(slc, tuple):
        assert t.start <= slc[0] and t.end >= slc[1]
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



if __name__=="__main__":
    text = '''\
-                    (A0*      
-                       *)     
play                  (V*)     
-                    (A1*)     
-                       *      '''
    text = text.split("\n")
    ret = get_predicate_arguments(text)
    print ret
