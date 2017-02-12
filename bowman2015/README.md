A large annotated corpus for learning natural language inference
================================================================

An tensorflow implementation of Bowman et al. (2015) - A large annotated corpus for learning natural language inference.
Network is not almost identical to the one described in their paper.

## Implementation Notes
* for the bidirectional LSTM,the correct way of accessing the encoded result
 is using `outputs[0]` and `outputs[-1]`, rather than just `outputs[-1]`
 (like some so-called Tensorflow-Tutorial does).

## Result

|Model  | dev    | test   |
|------|-------|-------|
| BiLSTM (100d) + glove.840W.300d (fixed) + MLP (relu) | 0.7981 | 0.7861 |
| 2 * BiLSTM (100d) + glove.840W.300d (fixed) + MLP (relu) | 0.7969 | 0.7867 |