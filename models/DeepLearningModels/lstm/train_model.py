
from keras import models, optimizers, Input
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Flatten, Embedding, concatenate, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras import regularizers

from lstm.preprocess_new import preProcessing 
import pickle
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score

# from lstm.test_model import testModel 
from lstm.check_model_results import testModel 

from keras.utils import np_utils


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
		encoded_claims = LSTM(32, return_sequences=True)(embed_claims)
		encoded_claims = LSTM(16)(encoded_claims)
		
		sentences_input = Input(shape=(None,), dtype='int32', name='sentences')
		embed_sents = Embedding(vocab_size_of_sents, embed_dim_s)(sentences_input)
		encoded_sentences = LSTM(32, return_sequences=True)(embed_sents)
		encoded_sentences = LSTM(16)(embed_sents)


		concatenate_layers = concatenate([encoded_claims, encoded_sentences],
											axis=-1)

		# concatenate_layers = Dropout(0.4)(concatenate_layers)
		concatenate_layers = Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu')(concatenate_layers)
		concatenate_layers = Dropout(0.5)(concatenate_layers)

		if nb_classes == 3:

			pred_label = Dense(3, activation='softmax')(concatenate_layers)

		else:

			pred_label = Dense(1, activation='sigmoid')(concatenate_layers)



		return claims_input, sentences_input, pred_label


if __name__ == '__main__':
	

	preprocess = preProcessing()
	training = TrainModel()
	metrics = Metrics()

	# fever_3 max len claim 65, 138
	# fever_3 claim word index 	7401 vocab size 16339
	# fever rej has max length claim   23, max length  128, claim vocab 5226, sents vocab 13567
	# birth place has claim vocab, sent vocab = 3384, 14617
	# institution has max claim length: 16, max sents length: 1140, claim vocab 15319, sents vocab 56717 
	# feversup claim vocab size 5462, sent vocab size 13908, claim max length 65 and sent max length 138
	
	# this vocab size is good for fever_3, fever_binary and fever_rej
	vocab_size_of_claims = 7000
	vocab_size_of_sents = 16339

	max_claims_length = 65
	max_sents_length = 150

	embedding_dim = 200

	train_dataset_name = 'fever_3'
	train_data = pickle.load(open("./lstm/datasets/train_"+str(train_dataset_name)+".pkl", "rb"))
	test_data = pickle.load(open("./lstm/datasets/test_"+str(train_dataset_name)+".pkl", "rb"))

	print ("train data ", train_data)
	# if dataset_name == 'birth_place':

		# claims are inside list so just remove them from list
	# train_data["claim"]= train_data["claim"].apply(lambda x: ','.join(map(str, x)))


	if train_dataset_name == "fever_sup":
		x_claim, x_sents, x_labels, x_claims_word_index,  x_sents_word_index, y_claims_data, y_sents_data, y_labels = preprocess.to_padding(train_data["claim"], train_data["sentence"], train_data["lablel"], test_data["claim"], test_data["sentence"], test_data["lablel"], 
						vocab_size_of_claims, vocab_size_of_sents, max_claims_length, max_sents_length)

	else:
		x_claim, x_sents, x_labels, x_claims_word_index,  x_sents_word_index, y_claims_data, y_sents_data, y_labels = preprocess.to_padding(train_data["claim"], train_data["sentence"], train_data["label"], test_data["claim"], test_data["sentence"], test_data["label"], 
					vocab_size_of_claims, vocab_size_of_sents, max_claims_length, max_sents_length)

	nb_classes = 3

	if nb_classes == 3:

		x_labels = np_utils.to_categorical(x_labels, nb_classes)
		loss = 'categorical_crossentropy'
	else:
		loss = 'binary_crossentropy'


	embeddings_index = preprocess.parse_glove(embedding_dim)
	pickle.dump(embeddings_index, open("/scratch/kkuma12s/embeddings_"+str(embedding_dim)+"dim_dict.pkl", "wb"))
	
	embeddings_index = pickle.load(open("/scratch/kkuma12s/embeddings_"+str(embedding_dim)+"dim_dict.pkl", "rb"))
	(embed_matrix_c, embed_dim_c) = preprocess.create_embedding_matrix(vocab_size_of_claims, embeddings_index, x_claims_word_index, embedding_dim)
	(embed_matrix_s, embed_dim_s) = preprocess.create_embedding_matrix(vocab_size_of_sents, embeddings_index, x_sents_word_index, embedding_dim)

	claims_input, sentences_input, pred_label = training.lstm_model(vocab_size_of_claims, vocab_size_of_sents, max_claims_length, max_sents_length, 
								embed_dim_c, embed_dim_s, nb_classes)


	model = Model([claims_input, sentences_input], pred_label)
	print (model.summary())

	model.layers[1].set_weights([embed_matrix_c])
	model.layers[1].trainable = False
	model.layers[4].set_weights([embed_matrix_s])
	model.layers[4].trainable = False

	early_stopping = EarlyStopping(monitor='val_loss', patience=2)
	model.compile(optimizer=optimizers.Adam(), loss=loss, metrics=['accuracy'])

	model_path = 'fever_simple_claims_model/model_lstm_fever_3_.h5'

	history = model.fit({'claims': x_claim, 'sentences': x_sents}, x_labels, 
								epochs=40, batch_size=64, validation_split=0.1, callbacks=[early_stopping, metrics,
												ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)])	


	test_lstm_model = testModel(y_claims_data, y_sents_data, y_labels, model_path, nb_classes)
	test_lstm_model.get_results_on_test_data(preprocess)