import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import numpy as np
import argparse

import jsonlines
import tensorflow as tf
import pandas as pd
import gzip
import keras
from keras import models, optimizers, Input, regularizers
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Flatten, Embedding, concatenate, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, CSVLogger
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.utils import np_utils

from bert.test_model import testModel 

config = tf.ConfigProto( device_count = {'GPU': 1} ) 
#config = tf.ConfigProto(log_device_placement=True)
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

class Metrics(Callback):

	def on_train_begin(self, logs={}):

		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []
	 
	def on_epoch_end(self, epoch, logs={}):

		val_predict = (np.asarray(model.predict([claims_sents_vec]))).round()
		val_targ = labels
		_val_f1 = f1_score(val_targ, val_predict, average = 'weighted')
		_val_recall = recall_score(val_targ, val_predict, average ='weighted')
		_val_precision = precision_score(val_targ, val_predict,average = 'weighted')
		self.val_f1s.append(_val_f1)
		self.val_recalls.append(_val_recall)
		self.val_precisions.append(_val_precision)
		print (' — val_f1: %f — val_precision: %f — val_recall %f' %( _val_f1, _val_precision, _val_recall))
		return

class TrainModel:


    def lstm_model(self, claim_length, embedding_dim, nb_classes):


        claims_input = Input(shape=(claim_length, embedding_dim), dtype='float32', name='claims')
        encoded_claims = LSTM(128, return_sequences=True)(claims_input)
        encoded_claims = LSTM(128, dropout=0.5, recurrent_dropout=0.4)(encoded_claims)
        # encoded_claims = LSTM(128, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)(encoded_claims)
        encoded_claims = Flatten()(claims_input)
        encoded_claims = Dense(256, kernel_regularizer=regularizers.l2(0.001), activation='relu')(encoded_claims)
        encoded_claims = Dropout(0.2)(encoded_claims)
        encoded_claims = Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu')(encoded_claims)
        encoded_claims = Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu')(encoded_claims)
        # encoded_claims = Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu')(encoded_claims)
        encoded_claims = Dropout(0.6)(encoded_claims)

        if nb_classes > 1:
            pred_label = Dense(nb_classes, activation='softmax')(encoded_claims)
        else:
            pred_label = Dense(1, activation='sigmoid')(encoded_claims)

        return claims_input, pred_label


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='start training')

    parser.add_argument('task', choices=['sent_retrieval','claim_classification'], 
                    help="what task should be performed?")

    task = parser.parse_args().task

    training = TrainModel()
    metrics = Metrics()

    max_seq_length = 256

    if task == 'sent_retrieval': 
        nb_classes = 2     
        dataset_name = "fever_full_binary_train"
        test_dataset_name = "fever_full_binary_dev"
        train_data = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/"+dataset_name+".jsonl" 

        dataset_name =='fever_full_binary_train'

        with gzip.open("/scratch/kkuma12s/new_embeddings/"+str(dataset_name)+"_bert_embeddings"+".pgz", 'rb') as f:
            claims_sents_vec = pickle.load(f)

    elif task == 'claim_classification':

        print ("claim classification selected ")
        nb_classes = 3

        train_dataset_name = "fever_full_binary_train_claim_labelling"
        test_dataset_name = "fever_full_binary_dev_claim_labelling"
        train_data = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/claim_classification/"+train_dataset_name+".jsonl" 
        test_data = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/"+test_dataset_name+".jsonl"

        with gzip.open("/scratch/kkuma12s/new_embeddings/fever_full_binary_train_bert_embeddings"+".pgz", 'rb') as f:
                claims_sents_vec = pickle.load(f)
    
    claims = []
    sents = []
    labels = []

    with jsonlines.open(train_data, mode='r') as f:
        tmp_dict = {}
        for example in f:
            claims.append(example["claim"])
            sents.append(example["sentence"])
            labels.append(example["label"])

        tmp_dict = {'claim':claims, 'sentence':sents, 'label':labels}
        train_data = pd.DataFrame(data=tmp_dict)


    if nb_classes > 1:
        labels = np_utils.to_categorical(train_data["label"], nb_classes)
        loss = 'categorical_crossentropy'

    else:
        loss = 'binary_crossentropy'

    embedding_dim = 768
    claims_input, pred_label = training.lstm_model(max_seq_length, embedding_dim, nb_classes)

    model = Model([claims_input], pred_label)
    print (model.summary())

    print ("labels ndim ", labels.shape)

    print (claims_sents_vec.shape)

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.compile(optimizer=optimizers.Adam(), loss=loss, metrics=['accuracy'])
    csv_logger = CSVLogger("bert_training.log")
    model_path = 'models_fever_full/claim_classifier_models/model_bert_fever_full_binary_claim_classifierAcc74.h5'
    history = model.fit({'claims': claims_sents_vec}, labels, 
                                epochs=60, batch_size=64, validation_split=0.12, callbacks=[early_stopping, metrics, csv_logger,
                                                ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)])
    dataset_name = test_dataset_name
    test_lstm_model = testModel(dataset_name, model_path, nb_classes)
    test_lstm_model.get_results_on_test_data()