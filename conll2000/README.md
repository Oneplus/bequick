Maxent chunker
==============

A demo code for using `sklearn` with maxent chunking.

## Data format

like the [crfsuite format](https://github.com/chokkan/crfsuite) with instance separated by `newline` and `label` is placed at the first column. An example is like:

```
B-NP    w[0]=Rockwell   w[1]=International      w[2]=Corp.
I-NP    w[-1]=Rockwell  w[0]=International      w[1]=Corp.
I-NP    w[-2]=Rockwell  w[-1]=International     w[0]=Corp.
B-NP    w[-2]=International     w[-1]=Corp.     w[0]='s
I-NP    w[-2]=Corp.     w[-1]='s        w[0]=Tulsa
I-NP    w[-2]='s        w[-1]=Tulsa     w[0]=unit

B-VP    w[-2]=Tulsa     w[-1]=unit      w[0]=said
B-NP    w[-2]=unit      w[-1]=said      w[0]=it
B-VP    w[-2]=said      w[-1]=it        w[0]=signed
B-NP    w[-2]=it        w[-1]=signed    w[0]=a
```

## Running CoNLL00 Chunking Task

Things has be prepared in script `pipeline.sh`. Just run `bash pipeline.sh` to get the result.

## Result

| Test | Tagging P | F-score |
|-----|----------|--------|
|      | 95.69 | 92.79 |