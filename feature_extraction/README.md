* ```feature_extractor.py``` is main file that collects features from all the methods
* ```feature_core.py``` Assign predicted label based on the scores returned by baselines: ```vector_space.py```, 
        ```tfidf.py```, and ```wmd.py``` 
* ```tune_model_param.py``` computes (precision, recall, f1 score) between predicted and actual labels
* ```train_test_sent_retrieval.py and train_test_claim_classification.py``` are used to train and test the sent retrieval as well as claim classifier independently
* ```test_complete_pipeline.py``` combines the results of sentence retrieval and claim classifier to determine the final output.
