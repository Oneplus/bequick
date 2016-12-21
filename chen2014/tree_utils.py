#!/usr/bin/env python
try:
    from .instance_builder import InstanceBuilder as IB
except (ValueError, SystemError) as e:
    from instance_builder import InstanceBuilder as IB


def is_projective(data):
    """

    :param data: list, with pseudo root at 0
    :return:
    """
    for payload in data[1:]:
        hed, mod = payload[IB.HED], payload[IB.ID]
        if hed < mod:
            for payload2 in data[hed + 1: mod]:
                hed2 = payload2[IB.HED]
                if hed2 < hed or hed2 > mod:
                    return False
        else:
            for payload2 in data[mod + 1: hed]:
                hed2 = payload2[IB.HED]
                if hed2 < mod or hed2 > hed:
                    return False
    return True


def is_tree(data):
    """

    :param data: list, with pseudo root at 0
    :return:
    """
    n = len(data)
    tree = [[] for _ in range(n + 1)]  # counting 0
    visited = [False for _ in range(n + 1)]
    for payload in data[1:]:
        hed = payload[IB.HED]
        tree[hed].append(payload[IB.ID])

    def travel(root):
        if visited[root]:
            return False   # loop or DAG

        visited[root] = True
        for c in tree[root]:
            if not travel(c):
                return False
        return True

    return travel(0)  # 0 is root by default.


def is_projective_raw(data):
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


def is_tree_raw(data):
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