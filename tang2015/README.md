Document Modeling with Gated Recurrent Neural Network for Sentiment Classification
==================================================================================

## Model

* FlattenAverage (`flat_avg`): flatten the words in the document and use the average as document representation.
* FlattenBiGRU (`flat_bigru`): Similar to FlattenBiLSTM but use Bi-GRU to represent the document.
* TreeAveragePipeGRU (`tree_avg_gru`):
* TreeGRUPipeAverage (`tree_gru_avg`):
* TreeGRUPipeGRU (`tree_gru_gru`):

## Experiments

On YELP dataset from Zhang et al. (2014), results are shown below

| Model                       | Dev   | Test  |
|-----------------------------|-------|-------|
| flat_avg + fix emb          | 49.46 | 49.57 |
| flat_avg + tune emb         | 61.03 | 60.89 |
| flat_bigru + fix emb        |       |       |
| flat_bigru + tune emb       |       |       |
| tree_avg_bigru + fix emb    |       |       |
| tree_bigru_avg  + fix emb   |       |       |
| tree_bigru_bigru + fix emb  | 66.03 | 65.70 |
| tree_bigru_gru + fix emb    | 65.89 | 65.51 |
| tree_avg_bigru + tune emb   | 61.03 | 60.89 |
| tree_bigru_avg  + tune emb  |  |  |
| tree_bigru_bigru + tune emb |  |  |
