Calibrating Features for Semantic Role Labeling
===============================================

This repo a reproduction of Xue and Palmer (2004), 
_Calibrating Features for Semantic Role Labeling_.
It uses `sklearn.linear_model.LogisticRegression` as an alternative
of the original maxentropy model.

## Prerequest

* Python 2.7
* scikit-learn: 0.15.2

## Data Preparation

* CoNLL2005 Shared Task Data
```
mkdir data
cd data
wget http://www.lsi.upc.edu/~srlconll/conll05st-release.tar.gz
tar zxvf conll05st-release.tar.gz
```

* Penn Treebank 3 (LDC1999T42)
```
scp /path/to/your/LDC1999T42.TGZ data/ldc1999t42.tgz
cd data
tar zxvf ldc1999t42.tgz
```

* Adapt the PTB3 WSJ to PTB2
```

```

## Run and Evaluate

```
./learn.sh
./devel.sh > sys.devel
zcat data/conll05st-release/devel/props/devel.24.props.gz > corr.devel
./srl-eval.pl sys.devel corr.devel
```

## Result