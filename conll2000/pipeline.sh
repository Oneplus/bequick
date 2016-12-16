#!/bin/bash

echo "download data."
if [ -f train.txt.gz ]; then
    echo "- train.txt.gz exists"
else
    curl -O http://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz
fi
if [ -f test.txt.gz ]; then
    echo "- test.txt.gz exisits"
else
    curl -O http://www.cnts.ua.ac.be/conll2000/chunking/test.txt.gz
fi

echo "extract feature."
if [ -f train.crfsuite.txt.gz ]; then
    echo "- train.crfsuite.txt.gz exists"
else
    zcat train.txt.gz | python ./chunking.py > train.crfsuite.txt
    gzip train.crfsuite.txt
fi
if [ -f test.crfsuite.txt.gz ]; then
    echo "- test.crfsuite.txt.gz exists"
else
    zcat test.txt.gz | python ./chunking.py > test.crfsuite.txt
    gzip test.crfsuite.txt
fi

echo "learning model."
python --version
time python ./run.py learn -f train.crfsuite.txt.gz -m logres.model
time python ./run.py test  -f test.crfsuite.txt.gz  -m logres.model
