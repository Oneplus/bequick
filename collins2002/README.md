Discriminative Training Methods for Hidden Markov Models
========================================================

It's an implementation of the Collins (2002), _Discriminative Training Methods for Hidden Markov Models_. It use numpy array as feature vector.

## Prerequisite
* python 2.7
* numpy

## Data Preparation

* postag data
The raw data should be in postag format(.pos) which has one instance each line. Form and postag are separated by backslash. The following is an example.

```
The/DT decision/NN was/VBD announced/VBN after/IN trading/NN ended/VBD ./.
```
* Penn Treebank 3 to .pos
```
mkdir data/
scp /path/to/your/LDC1999T42.TGZ data/ldc1999t42.tgz
cd data
tar zxvf ldc1999t42.tgz
cp ./wsj_convert.* data/LDC1999T42/TAGGED/POS/WSJ/
cd data/LDC1999T42/TAGGED/POS/WSJ/
./wsj_convert.sh
mv *.pos ../../../../
```

## Run and Evaluate

```
./run.sh
```

## Result

