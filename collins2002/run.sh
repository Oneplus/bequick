#!/bin/bash

#./extract.py ./wsj.train.pos > ./wsj.train.pos.crfsuite.txt
#./extract.py ./wsj.devel.pos > ./wsj.devel.pos.crfsuite.txt
#./extract.py ./wsj.test.pos  > ./wsj.test.pos.crfsuite.txt

./collins.py learn -t wsj.train.pos.crfsuite.txt \
    -d wsj.devel.pos.crfsuite.txt \
    -m wsj.pos.collins.model \
    -i 3

./collins.py tag -d wsj.test.pos.crfsuite.txt \
    -m wsj.pos.collins.model > wsj.pos.collins.output

awk '{print $1}' wsj.test.pos.crfsuite.txt | ./eval.py - wsj.pos.collins.output

#if [[ -n $(which crfsuite) ]]; then
#    echo "crfsuite benchmark is suspend because not toolkit is found."
#else
#    crfsuite learn -m wsj.pos.crfsuite.model \
#        -a l2sgd \
#        wsj.train.pos.crfsuite.txt
#    crfsuite tag -m wsj.pos.crfsuite.model \
#        wsj.test.pos.crfsuite.txt > wsj.pos.crfsuite.output
#    awk '{print $1}' wsj.test.pos.crfsuite.txt | ./eval.py - wsj.pos.crfsuite.output
#fi
