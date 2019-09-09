from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras import Input, regularizers
from keras.models import Model
from keras.layers import  Dense, Dropout, Embedding, concatenate, Activation, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import pickle 
from keras.optimizers import RMSprop, Adam
from elmo.preprocessing import preProcessing 

from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

claim_word_train_index = None
sents_word_train_index = None

def lstm_model(x_claim, x_sents, x_labels, test_claims_data, test_sents_data, test_labels):

	claim_length = 65
	sents_length = 138
	embedding_dim = 768
	nb_classes = 1

	preprocess = preProcessing()

	claims_input = Input(shape=(claim_length, embedding_dim), dtype='float32', name='claims')
	encoded_claims = LSTM({{choice([8, 16, 64, 128, 256, 512, 1024])}}, return_sequences=True, recurrent_dropout={{uniform(0, 1)}}, dropout={{uniform(0, 1)}})(claims_input)
	encoded_claims = LSTM({{choice([8, 16, 64, 128, 256, 512, 1024])}}, recurrent_dropout={{uniform(0, 1)}}, dropout={{uniform(0, 1)}})(encoded_claims)

	sentences_input = Input(shape=(sents_length, embedding_dim), dtype='float32', name='sentences')
	encoded_sentences = LSTM({{choice([8, 16, 64, 128, 256, 512, 1024])}}, return_sequences=True, recurrent_dropout={{uniform(0, 1)}}, dropout={{uniform(0, 1)}})(sentences_input)
	encoded_sentences = LSTM({{choice([16, 64, 256, 128, 512, 1024])}}, recurrent_dropout={{uniform(0, 1)}}, dropout={{uniform(0, 1)}})(encoded_sentences)


	concatenate_layers = concatenate([encoded_claims, encoded_sentences],
										axis=-1)

	concatenate_layers = Dropout({{uniform(0, 1)}})(concatenate_layers)
	concatenate_layers = Dense({{choice([8, 16, 32, 64, 256, 512, 1024])}}, kernel_regularizer=regularizers.l2(0.001), activation='relu')(concatenate_layers)
	concatenate_layers = Dense({{choice([8, 16, 32, 64, 256, 512, 1024])}}, kernel_regularizer=regularizers.l2(0.001), activation='relu')(concatenate_layers)
	# concatenate_layers = Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu')(concatenate_layers)

	concatenate_layers = Dropout({{uniform(0, 1)}})(concatenate_layers)
	if nb_classes == 3:

		pred_label = Dense(3, activation='softmax')(concatenate_layers)

	else:

		pred_label = Dense(1, activation='sigmoid')(concatenate_layers)

	model = Model([claims_input, sentences_input], pred_label)
	early_stopping = EarlyStopping(monitor='val_loss', patience=2)
	checkpointer = ModelCheckpoint(filepath='bert_keras_weights_feverbin.h5', 
	                               verbose=1, 
	                               save_best_only=True)

	model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

	model.fit({'claims': x_claim, 'sentences': x_sents}, x_labels,
	          batch_size={{choice([64, 128])}},
	          epochs=20,
	          verbose=2,
	          validation_split=0.1, callbacks=[early_stopping, checkpointer])



	score, acc = model.evaluate({'claims': test_claims_data, 'sentences': test_sents_data}, test_labels, verbose=0)
	print('Test accuracy:', acc)
	return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def data():


	preprocess = preProcessing()
	max_claims_length = 65
	max_sents_length = 138

	dataset_name = 'fever_sup'
	nb_classes = 1
	train_data = pickle.load(open("./bert/datasets/train_"+str(dataset_name)+".pkl", "rb"))

	x_claim = pickle.load(open("./bert/new_embeddings/train_claims_"+str(dataset_name)+"_embed"+".pkl", "rb"))
	x_sents = pickle.load(open("./bert/new_embeddings/train_sents_"+str(dataset_name)+"_embed"+".pkl", "rb"))
	
	x_labels = train_data["lablel"]

	if dataset_name == 'fever_3':
		x_labels = np_utils.to_categorical(x_labels, nb_classes)

	test_data = pickle.load(open("./bert/datasets/test_"+str(dataset_name)+".pkl", "rb"))

	test_claims_data = pickle.load(open("./bert/new_embeddings/test_claims_"+str(dataset_name)+"_embed"+".pkl", "rb"))
	test_sents_data = pickle.load(open("./bert/new_embeddings/test_sents_"+str(dataset_name)+"_embed"+".pkl", "rb"))
	
	test_labels = test_data["lablel"]

	if dataset_name == 'fever_3':
		test_labels = np_utils.to_categorical(test_labels, nb_classes)
		
	# x_claim, x_sents, x_labels = preprocess.to_padding(claim_embeddings, sents_embeddings, labels, max_claims_length, max_sents_length)
	# test_claims_data, test_sents_data, test_labels = preprocess.to_padding(test_claim_embeddings, test_sents_embeddings, test_labels, max_claims_length, max_sents_length)


	return (x_claim, x_sents, x_labels, test_claims_data, test_sents_data, test_labels)


if __name__ == '__main__':
	

	# metrics = Metrics()

	# print (len(claim_word_index)+1)
	# print (len(sents_word_index)+1)


	# claims_input, sentences_input, pred_label = training.lstm_with_fever_rej(vocab_size_of_claims, vocab_words_in_sents,
	# 																max_claims_length, max_sents_length, embed_dim_c, embed_dim_s)

	x_claim, x_sents, x_labels, test_claims_data, test_sents_data, test_labels = data()

	# results = lstm_model(x_claim, x_sents, x_labels, test_claims_data, test_sents_data, test_labels, 
	# 					claim_word_train_index, sents_word_train_index, claim_word_test_index, sents_word_test_index)


	# print (x_claim.shape)
	# print (x_sents.shape)
	# print (x_labels.shape)
	# print (test_claims_data.shape)
	# print (test_sents_data.shape)
	# print (test_labels.shape)

	best_run, best_model = optim.minimize(model=lstm_model,
	                                  data=data,
	                                  algo=tpe.suggest,
	                                  max_evals=5,
	                                  trials=Trials())

	print ("best model ", best_model.evaluate({'claims': test_claims_data, 'sentences': test_sents_data}, test_labels, verbose=0))
	print ("best run ", best_run)