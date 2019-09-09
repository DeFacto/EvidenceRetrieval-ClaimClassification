import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import numpy as np
import argparse

import keras
import tensorflow as tf
from keras import models, optimizers, Input
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Flatten, Embedding, concatenate, LSTM, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, CSVLogger
from keras import regularizers
import pandas as pd
import jsonlines
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.utils import np_utils

from elmo.preprocessing import preProcessing 
from elmo.test_model import testModel 


config = tf.ConfigProto( device_count = {'GPU': 0} ) 
#config = tf.ConfigProto(log_device_placement=True)
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

class Metrics(Callback):

	def on_train_begin(self, logs={}):

		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []
	 
	def on_epoch_end(self, epoch, logs={}):

		val_predict = (np.asarray(model.predict([claims_data, sents_data]))).round()
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


	def lstm_model(self, claim_length, sents_length, embedding_dim, nb_classes):


		claims_input = Input(shape=(claim_length, embedding_dim), dtype='float32', name='claims')
		encoded_claims = LSTM(128, return_sequences=True)(claims_input)
		encoded_claims = BatchNormalization()(encoded_claims)
		encoded_claims = LSTM(512)(encoded_claims)
		# encoded_claims = LSTM(512)(encoded_claims)
		# encoded_claims = Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu')(encoded_claims)
		# encoded_claims = Dropout(0.5)(encoded_claims)
		encoded_claims = BatchNormalization()(encoded_claims)
		sentences_input = Input(shape=(sents_length, embedding_dim), dtype='float32', name='sentences')
		encoded_sentences = LSTM(512, dropout=0.3, return_sequences=True)(sentences_input)
		encoded_sentences = BatchNormalization()(encoded_sentences)
		encoded_sentences = LSTM(512)(encoded_sentences)
		# encoded_sentences = Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu')(encoded_sentences)
		# encoded_sentences = BatchNormaslization()(encoded_sentences)
		# encoded_claims = Dropout(0.5)(encoded_claims)
		concatenate_layers = concatenate([encoded_claims, encoded_sentences],
											axis=-1)

		# concatenate_layers = Dropout(0.1)(concatenate_layers)
		# concatenate_layers = Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu')(concatenate_layers)
		# concatenate_layers = Dropout(0.6)(concatenate_layers)
		# concatenate_layers = Dense(16, activation='relu')(concatenate_layers)
		concatenate_layers = Dense(128, activation='relu')(concatenate_layers)
		concatenate_layers = BatchNormalization()(concatenate_layers)
		concatenate_layers = Dropout(0.6)(concatenate_layers)

		if nb_classes > 1 :

			pred_label = Dense(nb_classes, activation='softmax')(concatenate_layers)

		else:

			pred_label = Dense(nb_classes, activation='sigmoid')(concatenate_layers)



		return claims_input, sentences_input, pred_label


	def read_dataset(self, data):

		claims = []
		sents = []
		labels = []

		with jsonlines.open(data, mode='r') as f:

			for example in f:
				# claims.append(self.text_to_wordlist(example["claim"]))
				claims.append(example["claim"])
				# sents.append(self.text_to_wordlist(example["sentence"]))
				sents.append(example["sentence"])
				labels.append(example["label"])
				#sentence

			tmp_dict = {'claim':claims, 'sentence':sents, 'label':labels}
			train_data = pd.DataFrame(data=tmp_dict)

		return train_data


if __name__ == '__main__':
	

	parser = argparse.ArgumentParser(description='start training')

	parser.add_argument('task', choices=['sent_retrieval','claim_classification'], 
					help="what task should be performed?")

	task = parser.parse_args().task

	preprocess = preProcessing()
	training = TrainModel()
	metrics = Metrics()

	max_claims_length = 40
	max_sents_length = 250

	if task == 'sent_retrieval':	
		nb_classes = 2
		train_dataset_name = "fever_full_binary_train"
		test_dataset_name = "fever_full_binary_dev"
		train_data = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/"+train_dataset_name+".jsonl" 

	elif task == 'claim_classification':

		print ("claim classification selected ")
		nb_classes = 3
		train_dataset_name = "fever_full_binary_train_claim_labelling"
		test_dataset_name = "fever_full_binary_dev_claim_labelling"
		train_data = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/claim_classification/"+train_dataset_name+".jsonl" 
		test_data = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/claim_classification/"+test_dataset_name+".jsonl"

	
	train_data = training.read_dataset(train_data)

	#embeddings of sent retrieval and claim classification are same
	embeddings_name = "fever_full_binary_train"
	claim_embeddings = pickle.load(open("/scratch/kkuma12s/elmo_embeddings/train_claim_elmo_emb_"+str(embeddings_name)+".pkl", "rb"))
	sents_embeddings = pickle.load(open("/scratch/kkuma12s/elmo_embeddings/train_sents_elmo_emb_"+str(embeddings_name)+".pkl", "rb"))
	labels = train_data["label"]

	claims_data, sents_data = preprocess.to_padding(claim_embeddings, sents_embeddings, labels, max_claims_length, max_sents_length)

	print ("claims data shape ", claims_data.shape)

	if nb_classes > 1:
		labels = np.asarray(labels)
		labels = np_utils.to_categorical(labels, nb_classes)
		loss = 'categorical_crossentropy'

	else:
		loss = 'binary_crossentropy'

	embedding_dim = 1024
	claims_input, sentences_input, pred_label = training.lstm_model(max_claims_length, max_sents_length, embedding_dim, nb_classes)
	model = Model([claims_input, sentences_input], pred_label)
	print (model.summary())

	early_stopping = EarlyStopping(monitor='val_loss', patience=1)
	model.compile(optimizer=optimizers.Adam(), loss=loss, metrics=['accuracy'])
	
	model_path = 'model_elmo_fever_full_binary_sent_ret.h5'
	
	csv_logger = CSVLogger('elmo_training.log')

	print ("claims shape ", claims_data.shape)
	print ("sents data shape ", sents_data.shape)

	history = model.fit({'claims': claims_data, 'sentences': sents_data}, labels, 
								epochs=40, batch_size=64, validation_split=0.1, callbacks=[early_stopping, metrics, csv_logger,
												ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)])	
	
	test_lstm_model = testModel(max_claims_length, max_sents_length, model_path, nb_classes, dataset_name=test_dataset_name)
	test_lstm_model.get_results_on_test_data(preprocess)