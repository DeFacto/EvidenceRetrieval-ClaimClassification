import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import random
import pickle
import numpy as np
import argparse
import re

import jsonlines
import tensorflow as tf
import keras
from keras import models, optimizers, Input
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Flatten, Embedding, concatenate, LSTM, Bidirectional, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, CSVLogger
from keras import regularizers
from keras.utils import np_utils
from sklearn.metrics import f1_score, precision_score, recall_score

from lstm.preprocess_new import preProcessing 
from lstm.test_model_new import testModel 

config = tf.ConfigProto(device_count = {'GPU': 2} )
# config = tf.ConfigProto("/device:GPU:0")

sess = tf.Session(config=config)
keras.backend.set_session(sess)

class Metrics(Callback):

	def on_train_begin(self, logs={}):

		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []
	 
	def on_epoch_end(self, epoch, logs={}):

		val_predict = (np.asarray(model.predict([x_claim, x_sents]))).round()
		val_targ = x_labels
		_val_f1 = f1_score(val_targ, val_predict, average = 'weighted')
		_val_recall = recall_score(val_targ, val_predict, average ='weighted')
		_val_precision = precision_score(val_targ, val_predict,average = 'weighted')
		self.val_f1s.append(_val_f1)
		self.val_recalls.append(_val_recall)
		self.val_precisions.append(_val_precision)
		print (' — val_f1: %f — val_precision: %f — val_recall %f' %( _val_f1, _val_precision, _val_recall))
		return


class TrainModel:


	def lstm_model(self, vocab_size_of_claims, vocab_size_of_sents, maxlen_claims, maxlen_sents, 
								embed_dim_c, embed_dim_s, nb_classes):


		claims_input = Input(shape=(None,), dtype='int32', name='claims')
		embed_claims = Embedding(vocab_size_of_claims, embed_dim_c)(claims_input)
		encoded_claims = LSTM(256, return_sequences=True)(embed_claims)
		encoded_claims = LSTM(16)(encoded_claims)
		encoded_claims = BatchNormalization()(encoded_claims)
		# encoded_claims = Dense(32, activation='relu')(encoded_claims)
		# encoded_claims = Dropout(0.5)(encoded_claims)
		# encoded_claims = BatchNormalization()(encoded_claims)
		# encoded_claims = Dense(256, kernel_regularizer=regularizers.l2(0.001), activation='relu')(encoded_claims)
		# encoded_claims = BatchNormalization()(encoded_claims)
		# encoded_claims = Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu')(encoded_claims)
		# encoded_claims = LSTM(64)(embed_claims)
		# encoded_claims = LSTM(256 , recurrent_dropout=0.2, dropout=0.2)(encoded_claims)
	
		sentences_input = Input(shape=(None,), dtype='int32', name='sentences')
		embed_sents = Embedding(vocab_size_of_sents, embed_dim_s)(sentences_input)
		encoded_sentences = LSTM(256, return_sequences=True)(embed_sents)
		encoded_sentences = LSTM(64)(encoded_sentences)
		encoded_sentences = BatchNormalization()(encoded_sentences)
		# encoded_sentences = Dense(32, activation='relu')(encoded_sentences)
		# encoded_sentences = Dropout(0.5)(encoded_sentences)
		# encoded_sentences = BatchNormalization()(encoded_sentences)
		# encoded_sentences = Dense(256, kernel_regularizer=regularizers.l2(0.001), activation='relu')(encoded_sentences)
		# encoded_sentences = BatchNormalization()(encoded_sentences)
		# encoded_sentences = Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu')(encoded_sentences)
		# encoded_sentences = LSTM(64, return_sequences=True)(embed_sents)
		# encoded_sentences = LSTM(16)(encoded_sentences)

		concatenate_layers = concatenate([encoded_claims, encoded_sentences],axis=-1)
		# concatenate_layers = Flatten()(concatenate_layers)
		concatenate_layers = Dropout(0.5)(concatenate_layers)
		concatenate_layers = Dense(64, activation='relu')(concatenate_layers)
		# concatenate_layers = BatchNormalization()(concatenate_layers)
		# concatenate_layers = Dropout(0.1)(concatenate_layers)


		if nb_classes == 3:

			pred_label = Dense(nb_classes, activation='softmax')(concatenate_layers)

		else:

			pred_label = Dense(1, activation='sigmoid')(concatenate_layers)



		return claims_input, sentences_input, pred_label


	def read_dataset(self, data):

		claims = []
		sents = []
		labels = []

		with jsonlines.open(data, mode='r') as f:
			for example in f:
				claims.append(self.text_to_wordlist(example["claim"]))
				# claims.append(example["claim"])
				sents.append(self.text_to_wordlist(example["sentence"]))
				# sents.append(example["sentence"])
				labels.append(example["label"])
				#sentence

		return claims, sents, labels


	def text_to_wordlist(self, text, remove_stopwords=False, stem_words=False):

	    # Clean the text, with the option to remove stopwords and to stem words.
	    
	    # Convert words to lower case and split them
	    text = text.lower().split()

	    # Optionally, remove stop words
	    if remove_stopwords:
	        stops = set(stopwords.words("english"))
	        text = [w for w in text if not w in stops]
	    
	    text = " ".join(text)

	    # Clean the text
	    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
	    text = re.sub(r"what's", "what is ", text)
	    text = re.sub(r"\'s", " ", text)
	    text = re.sub(r"\'ve", " have ", text)
	    text = re.sub(r"can't", "cannot ", text)
	    text = re.sub(r"n't", " not ", text)
	    text = re.sub(r"i'm", "i am ", text)
	    text = re.sub(r"\'re", " are ", text)
	    text = re.sub(r"\'d", " would ", text)
	    text = re.sub(r"\'ll", " will ", text)
	    text = re.sub(r",", " ", text)
	    text = re.sub(r"\.", " ", text)
	    text = re.sub(r"!", " ! ", text)
	    text = re.sub(r"\/", " ", text)
	    text = re.sub(r"\^", " ^ ", text)
	    text = re.sub(r"\+", " + ", text)
	    text = re.sub(r"\-", " - ", text)
	    text = re.sub(r"\=", " = ", text)
	    text = re.sub(r"'", " ", text)
	    # text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
	    text = re.sub(r":", " : ", text)
	    text = re.sub(r" e g ", " eg ", text)
	    text = re.sub(r" b g ", " bg ", text)
	    # text = re.sub(r" u s ", " american ", text)
	    text = re.sub(r"\0s", "0", text)
	    text = re.sub(r" 9 11 ", "911", text)
	    # text = re.sub(r"e - mail", "email", text)
	    text = re.sub(r"j k", "jk", text)
	    text = re.sub(r"\s{2,}", " ", text)
	    
	    # Optionally, shorten words to their stems
	    if stem_words:
	        text = text.split()
	        stemmer = SnowballStemmer('english')
	        stemmed_words = [stemmer.stem(word) for word in text]
	        text = " ".join(stemmed_words)
	    
	    # Return a list of words
	    return (text)


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='start training')

	parser.add_argument('task', choices=['sent_retrieval','claim_classification'], 
					help="what task should be performed?")

	task = parser.parse_args().task

	preprocess = preProcessing()
	train_model = TrainModel()
	metrics = Metrics()

	# fever_3 max len claim 65, 138
	# fever_3 claim word index 	7401 vocab size 16339
	# fever rej has max length claim   23, max length  128, claim vocab 5226, sents vocab 13567
	# birth place has claim vocab, sent vocab = 3384, 14617
	# institution has max claimm length: 16, max sents length: 1140, claim vocab 15319, sents vocab 56717 
	# feversup claim vocab size 5462, sent vocab size 13908, claim max length 65 and sent max length 138
	vocab_size_of_claims = 20000
	vocab_size_of_sents = 80000

	max_claims_length = 50
	max_sents_length = 220 #300
 
	embedding_dim = 300

	
	if task == 'sent_retrieval':

		nb_classes = 2
		train_dataset_name = "fever_full_binary_train"
		test_dataset_name = "fever_full_binary_dev"
		train_data = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/"+train_dataset_name+".jsonl" 
		test_data = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/"+test_dataset_name+".jsonl"

	elif task == 'claim_classification':
		nb_classes = 3
		train_dataset_name = "fever_full_binary_train_claim_labelling"
		test_dataset_name = "fever_full_binary_dev_claim_labelling"
		train_data = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/claim_classification/"+train_dataset_name+".jsonl" 
		test_data = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/claim_classification/"+test_dataset_name+".jsonl"


	x_claims, x_sents, x_labels = train_model.read_dataset(train_data)
	y_claims, y_sents, y_labels = train_model.read_dataset(test_data)


	print ("claims length ", len(x_claims))
	print ("sents length ", len(x_sents))
	print ("labels ", len(x_labels))

	x_claim, x_sents, x_labels, x_claims_word_index,  x_sents_word_index, y_claims_data, y_sents_data, y_labels = preprocess.to_padding(x_claims, x_sents, x_labels, y_claims, y_sents, y_labels, 
					vocab_size_of_claims, vocab_size_of_sents, max_claims_length, max_sents_length)
	
	print ("x claim word index ", len(x_claims_word_index))
	print ("x sent word index ", len(x_sents_word_index))
	
	

	if nb_classes == 3:

		x_labels = np_utils.to_categorical(x_labels, nb_classes)
		loss = 'categorical_crossentropy'
	else:
		loss = 'binary_crossentropy'

	# embeddings_index = preprocess.parse_glove(embedding_dim)
	# pickle.dump(embeddings_index, open("embeddings_"+str(embedding_dim)+"dim_dict.pkl", "wb"))
	
	embeddings_index = pickle.load(open("/scratch/kkuma12s/embeddings_"+str(embedding_dim)+"dim_dict.pkl", "rb"))

	(embed_matrix_c, embed_dim_c) = preprocess.create_embedding_matrix(vocab_size_of_claims, embeddings_index, x_claims_word_index, embedding_dim)
	(embed_matrix_s, embed_dim_s) = preprocess.create_embedding_matrix(vocab_size_of_sents, embeddings_index, x_sents_word_index, embedding_dim)

	print ("embed_matrix_c shape ", embed_matrix_c.shape)
	print ("embed_matrix_s shape ", embed_matrix_s.shape)

	claims_input, sentences_input, pred_label = train_model.lstm_model(vocab_size_of_claims, vocab_size_of_sents, max_claims_length, max_sents_length, 
								embed_dim_c, embed_dim_s, nb_classes)


	model = Model([claims_input, sentences_input], pred_label)
	print (model.summary())

	model.layers[2].set_weights([embed_matrix_c])
	model.layers[2].trainable = False
	model.layers[3].set_weights([embed_matrix_s])
	model.layers[3].trainable = False

	# model.layers[1].set_weights([embed_matrix_c])
	# model.layers[1].trainable = False
	# model.layers[4].set_weights([embed_matrix_s])
	# model.layers[4].trainable = False

	# model.layers[3].set_weights([embed_matrix_c])
	# model.layers[3].trainable = False
	# model.layers[2].set_weights([embed_matrix_s])
	# model.layers[2].trainable = False

	early_stopping = EarlyStopping(monitor='val_loss', patience=2)
	model.compile(optimizer=optimizers.Adam(), loss=loss, metrics=['accuracy'])

	csv_logger = CSVLogger('lstm_training_tf.log')
	model_path = 'model_lstm_fever_full_binary_claim_classifier.h5'
	history = model.fit({'claims': x_claim, 'sentences': x_sents}, x_labels, 
								epochs=60, batch_size=64, validation_split=0.12, callbacks=[early_stopping, metrics, csv_logger,
												ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)])	


	test_lstm_model = testModel(y_claims_data, y_sents_data, y_labels, model_path, test_dataset_name, nb_classes)
	test_lstm_model.get_results_on_test_data(preprocess)
	