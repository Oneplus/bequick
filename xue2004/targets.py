#!/usr/bin/env python


def get_target_indices(target):
    retval = []
    for idx, line in enumerate(target):
        if line.strip() != "-":
            retval.append(idx)
    return retval
