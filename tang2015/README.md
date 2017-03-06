Document Modeling with Gated Recurrent Neural Network for Sentiment Classification
==================================================================================

## Model

* FlattenAverage (`flat_avg`): flatten the words in the document and use the average as document representation.
* FlattenBiLSTM (`flat_bilstm`): flatten the words in the document and feed the word sequence into a Bi-LSTM to
  represent the document.
* FlattenBiGRU (`flat_bigru`): Similar to FlattenBiLSTM but use Bi-GRU to represent the document.
* TreeAveragePipeGRU (`tree_avg_gru`):
* TreeGRUPipeAverage (`tree_gru_avg`):
* TreeGRUPipeGRU (`tree_gru_gru`):

## Experiments

