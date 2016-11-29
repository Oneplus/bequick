#!/usr/bin/env python


def is_projective(data):
    for payload in data:
        hed, mod = payload['head'], payload['id']
        if hed < mod:
            for payload2 in data[hed: mod - 1]:
                hed2 = payload2['head']
                if hed2 < hed or hed2 > mod:
                    return False
        else:
            for payload2 in data[mod: hed - 1]:
                hed2 = payload2['head']
                if hed2 < mod or hed2 > hed:
                    return False
    return True


def is_tree(data):
    n = len(data)
    tree = [[] for _ in range(n + 1)]  # counting 0
    visited = [False for _ in range(n + 1)]
    for payload in data:
        hed = payload['head']
        tree[hed].append(payload['id'])

    def travel(root):
        if visited[root]:
            return False   # loop or DAG

        visited[root] = True
        for c in tree[root]:
            if not travel(c):
                return False
        return True

    return travel(0)  # 0 is root by default.
