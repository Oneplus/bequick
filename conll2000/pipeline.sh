#!/bin/bash

echo "download data."
wget http://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz
wget http://www.cnts.ua.ac.be/conll2000/chunking/test.txt.gz

echo "extract feature."
zcat train.txt.gz | ./chunking.py > train.crfsuite.txt
zcat test.txt.gz | ./chunking.py > test.crfsuite.txt

gzip train.crfsuite.txt
gzip test.crfsuite.txt

echo "learning model."
time ./main.py learn -f train.crfsuite.txt.gz -m logres.model
time ./main.py test  -f test.crfsuite.txt.gz  -m logres.model
