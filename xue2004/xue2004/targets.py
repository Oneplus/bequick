#!/usr/bin/env python
import re

def get_target_indices(target):
    retval = []
    for idx, line in enumerate(target):
        if line != "-":
            retval.append(idx)
    return retval

