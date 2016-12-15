Bidirectional LSTM for Sequence Labeling
====================================

A simple bidirectional LSTM sequence labeler. It can be used on Part-of-Speech tagging, Named Entity recognition and
other similar tasks.

## Implementation Notes
* `sequence_length` is very important for bi-LSTM in NLP. Since the sequence length varies significantly in NLP
   problems (some of the sentence has more than 100 words while the average sentence length is about 30), if we use
   zero-padding on such kind of data, the model would learn too much zero instance.

## Sanity Test

Test on several data sets.

| Dataset | Size |
|--------|------|
| CTB5.1  | #Train=16,111, #Devel=805, #Test=1,915 |

Results

| Dataset | Model | Learning | Devel | Test |
|--------|------|---------|------|-----|
| CTB5.1 | bi-LSTM / Concat / 2-layer MLP (ReLu) | Adam / batch=32 / iter=30 | 86.22 | 86.03 |
| CTB5.1 | bi-LSTM / Concat / 2-layer MLP (tanh) | Adam / batch=32 / iter=30 | 93.29 | 92.91 |
| CTB5.1 | bi-LSTM / Concat / 2-layer MLP (tanh) / CRF | Adam / batch=32 / iter=10 | 93.17 | 92.69 |
