#!/usr/bin/env python
__description__ = "Extract feature from .pos (postag format) file."
__author__ = "Yijia Liu"
__email__ = "oneplus.lau@gmail.com"

import sys
import re
try:
    fp = open(sys.argv[1])
except:
    fp = sys.stdin

for line in fp:
    tokens = line.strip().split()
    words = [token.rsplit("/", 1)[0] for token in tokens]
    postags = [token.rsplit("/", 1)[1] for token in tokens]

    L = len(words)
    for i in xrange(L):
        w_2 = words[i- 2] if i- 2 >= 0 else "__bos__"
        w_1 = words[i- 1] if i- 1 >= 0 else "__bos__"
        w0 = words[i]
        w1 = words[i+ 1] if i+ 1 < L else "__eos__"
        w2 = words[i+ 2] if i+ 2 < L else "__eos__"
        NUM = "true" if re.search("[0-9]", words[i]) is not None else "false"
        UPPER = "true" if re.search("[A-Z]", words[i]) is not None else "false"
        HYPEN = "true" if '-' in words[i] else "false"

        output = ["w[-2]=%s" % w_2,
                "w[-1]=%s" % w_1,
                "w[0]=%s" % w0,
                "w[1]=%s" % w1,
                "w[2]=%s" % w2,
                "num=%s" % NUM,
                "upper=%s" % UPPER,
                "hypen=%s" % HYPEN]

        if len(w0) >= 1:
            output.append("pre[1]=%s" % w0[:1])
            output.append("suf[1]=%s" % w0[-1:])

        if len(w0) >= 2:
            output.append("pre[2]=%s" % w0[:2])
            output.append("suf[2]=%s" % w0[-2:])

        if len(w0) >= 3:
            output.append("pre[3]=%s" % w0[:3])
            output.append("suf[3]=%s" % w0[-3:])

        print "%s\t%s" % (postags[i], "\t".join(output))
    print
