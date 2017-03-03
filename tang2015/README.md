Document Modeling with Gated Recurrent Neural Network for Sentiment Classification
==================================================================================

## Model

* FlattenAverage (`flat_avg`): flatten the words in the document and use the average as document representation.
* FlattenBiLSTM (`flat_bilstm`): flatten the words in the document and feed the word sequence into a Bi-LSTM to
  represent the document.
* FlattenBiGRU (`flat_bigru`): Similar to FlattenBiLSTM but use Bi-GRU to represent the document.
* DoctreeBiLSTM (`doctree_bilstm`):
* DoctreeAverage (`doctree_avg`): represent sentence with BiLSTM and average the sentence representation as
  the document representation.
* DoctreeBiLSTM (`doctree_bilstm`):