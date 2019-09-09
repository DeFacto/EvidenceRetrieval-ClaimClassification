from keras import models, optimizers, Input
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Flatten, Embedding, concatenate, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras import regularizers
from elmo.preprocessing import preProcessing 
import pickle
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from elmo.test_model import testModel 
from keras.utils import np_utils

# Script use for training on fever simple claims dataset

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
		encoded_claims = LSTM(32, return_sequences=True)(claims_input)
		encoded_claims = LSTM(16)(encoded_claims)
		# encoded_claims = LSTM(8)(claims_input)
		# encoded_claims = Dense(8, kernel_regularizer=regularizers.l2(0.001), activation='relu')(encoded_claims)
		# encoded_claims = LSTM(32, recurrent_dropout=0.5, dropout=0.5)(encoded_claims)
		sentences_input = Input(shape=(sents_length, embedding_dim), dtype='float32', name='sentences')
		encoded_sentences = LSTM(32, return_sequences=True)(sentences_input)
		encoded_sentences = LSTM(16)(encoded_sentences)
		# encoded_sentences = LSTM(16)(sentences_input)
		# # encoded_sentences = LSTM(32, recurrent_dropout=0.5, dropout=0.5)(encoded_sentences)
		# encoded_sentences = Dense(8, kernel_regularizer=regularizers.l2(0.001), activation='relu')(encoded_sentences)
		concatenate_layers = concatenate([encoded_claims, encoded_sentences],
											axis=-1)
		# concatenate_layers = Dropout(0.5)(concatenate_layers)
		concatenate_layers = Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu')(concatenate_layers)
		# concatenate_layers = Dropout(0.6)(concatenate_layers)
		# concatenate_layers = Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu')(concatenate_layers)
		# concatenate_layers = Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu')(concatenate_layers)

		concatenate_layers = Dropout(0.6)(concatenate_layers)

		if nb_classes == 3:
			pred_label = Dense(3, activation='softmax')(concatenate_layers)
		else:
			pred_label = Dense(1, activation='sigmoid')(concatenate_layers)

		return claims_input, sentences_input, pred_label

if __name__ == '__main__':
	
	preprocess = preProcessing()
	training = TrainModel()
	metrics = Metrics()

	max_claims_length = 65
	max_sents_length = 128

	dataset_name = 'fever_3'
	train_data = pickle.load(open("./elmo/datasets/train_"+str(dataset_name)+".pkl", "rb"))

	claim_embeddings = pickle.load(open("./elmo/embeddings/train_claim_elmo_emb_"+str(dataset_name)+".pkl", "rb"))
	sents_embeddings = pickle.load(open("./elmo/embeddings/train_sents_elmo_emb_"+str(dataset_name)+".pkl", "rb"))
	
	if dataset_name == 'fever_sup':
		labels = train_data["lablel"]

	else:
		labels = train_data["label"]

	print (len(claim_embeddings))
	print (len(sents_embeddings))
	print (len(labels))

	claims_data, sents_data = preprocess.to_padding(claim_embeddings, sents_embeddings, labels, max_claims_length, max_sents_length)

	# print (claims_data.ndim)
	# print (sents_data.ndim)
	# print (labels.ndim)
	
	nb_classes = 3

	if nb_classes == 3:
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
	model_path = 'fever_simple_claims_model/model_elmo_fever_3_.h5'
	history = model.fit({'claims': claims_data, 'sentences': sents_data}, labels, 
								epochs=40, batch_size=64, validation_split=0.1, callbacks=[early_stopping, metrics,
												ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)])	


	test_lstm_model = testModel(max_claims_length, max_sents_length, model_path, nb_classes, dataset_name=dataset_name)
	test_lstm_model.get_results_on_test_data(preprocess)
