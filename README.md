# EvidenceRetrieval-ClaimClassification

In this project, different methods for evidence retrieval and Claim Classification are evaluated. Specifically we used following methods:
* Vector Space
* TF-IDF
* Word Mover's Distance (WMD)
* Simple LSTM using word2vec
* Simple LSTM using ELMo
* Simple LSTM using BERT

To run classical methods such as Vector Space, TF-IDF, and WMD execute:
```
python -m feature_extraction.script_name
```
Example:
```
python -m feature_extraction.feature_extractor
```

To run deep learning methods, following steps are considered: [Deep Learning Methods](https://github.com/DeFacto/EvidenceRetrieval-ClaimClassification/tree/master/models/DeepLearningModels)
