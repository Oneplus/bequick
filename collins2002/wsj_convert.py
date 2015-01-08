#!/usr/bin/env python
import sys
import re
context = open(sys.argv[1], "r").read().strip()
context2 = ""
for line in context.split("\n"):
    if re.match("^=+$", line):
        continue
    context2 += line.strip().lstrip("[").rstrip("]").strip() + "\n"
context2 = context2.strip()
#print >> sys.stderr, context2
for instance in context2.split("\n\n"):
    if len(instance) == 0:
        continue
    output = []
    for line in instance.split("\n"):
        output.extend([token.rsplit("/", 1) for token in line.split()])
    try:
        print " ".join(["%s/%s" % (o[0], o[1]) for o in output])
    except:
        print >> sys.stderr, sys.argv[1]
