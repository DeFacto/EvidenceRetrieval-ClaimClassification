# For training and testing sentence retrieval and claim classification models on FEVER Original Set

Following script is used:

``` train_model_fever_full.py and test_model.py``` 

## Datasets used for training sentence retrieval and claim classification models

Dataset contains equal number of true +ves and -ves

``` data/fever-full/fever_full_binary_train.jsonl" and data/fever-full/claim_classification/fever_full_binary_train_claim_labelling.jsonl``` 


## Datasets used for testing sentence and claim classification models individually

```
data/fever-full/fever_full_binary_dev.jsonl" and data/fever-full/claim_classification/fever_full_binary_dev_claim_labelling.jsonl
```

# Testing sentence retrieval and claim classification models

Following script is used:
```
test_sent_retrieval_claim_cls_dev_set.py
```

# Combine claims having same ids

Following script is used:

```
test_complete_pipeline.py
```

# Majority vote classifier

```
get_final_results.py
```

# Obtain Bert Embeddings

* BERT embeddings were obtained using [getting-elmo-like-contextual-word-embedding](https://github.com/hanxiao/bert-as-service#getting-elmo-like-contextual-word-embedding)

* Model used for embeddings: ```BERT-Base, Multilingual Cased (New, recommended): 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters```
