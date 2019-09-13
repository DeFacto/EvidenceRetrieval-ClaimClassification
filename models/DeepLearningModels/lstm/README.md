# Spliting the Datasets

* Train and test datasets were created in the ratio of 0.9 and 0.1 respectively and validation ratio was done using k-fold validation. While for TFIDF, WMD and Vspace split ratio of train, validation and test set was 0.6, 0.2, 0.2 respectively

# Training and Testing on FEVER Original Set 

## For training and testing sentence retrieval and claim classification models on FEVER Original Set

Following script is used:

``` train_model_fever_full.py and check_model_results.py``` 

### Datasets used for training sentence retrieval and claim classification models

Dataset contains equal number of true +ves and -ves

``` data/fever-full/fever_full_binary_train.jsonl" and data/fever-full/claim_classification/fever_full_binary_train_claim_labelling.jsonl``` 


### Datasets used for testing sentence and claim classification models individually

```
data/fever-full/fever_full_binary_dev.jsonl" and data/fever-full/claim_classification/fever_full_binary_dev_claim_labelling.jsonl
```

## Testing sentence retrieval and claim classification models combined

Following script is used:
```
test_sent_retrieval_claim_cls_dev_set.py
```

## Testing sentence retrieval individually

Following script is used:
```
test_sent_retrieval_complete_dev.py
```

## Combine claims having same ids

Following script is used:

```
test_complete_pipeline.py
```

## Majority vote classifier

```
get_final_results.py
```

---

# Training and Testing on FEVER Simple Claims Set


## For training and testing claim classification models

Following script is used:

``` train_model.py and check_model_results.py``` 

---
