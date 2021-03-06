#!/usr/bin/env python
from __future__ import print_function
import sys


if sys.argv[1] == "-":
    fp1 = sys.stdin
else:
    try:
        fp1 = open(sys.argv[1], "r")
    except IOError:
        print("failed to open file.", file=sys.stderr)
        sys.exit(1)

if sys.argv[1] != "-" and sys.argv[2] == "-":
    fp2 = sys.stdin
else:
    try:
        fp2 = open(sys.argv[2], "r")
    except IOError:
        print("failed to open file.", file=sys.stderr)
        sys.exit(1)

references = [data.split("\n") for data in fp1.read().strip().split("\n\n")]
answers = [data.split("\n") for data in fp2.read().strip().split("\n\n")]

assert len(references) == len(answers), "number of instance is not equal: (%d,%d)" % (len(references), len(answers))

nr_instances = 0
nr_lines = 0
nr_corr, nr_tot = 0, 0 
for reference, answer in zip(references, answers):
    nr_instances += 1
    assert len(reference) == len(answer), "number of tags is not equal (%d, %d)" % (nr_instances, nr_lines)
    for r, a in zip(reference, answer):
        if r == a:
            nr_corr += 1
        nr_tot += 1
    nr_lines += (len(reference) + 1)
print("tagging accuracy = %f" % (float(nr_corr) / nr_tot))
