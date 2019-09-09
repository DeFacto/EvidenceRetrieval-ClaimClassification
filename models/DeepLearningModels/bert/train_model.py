from keras import models, optimizers, Input, regularizers
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Flatten, Embedding, concatenate, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score, precision_score, recall_score
from bert.test_model import testModel 
from keras.utils import np_utils

import pickle
import numpy as np

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
            encoded_claims = LSTM(512, return_sequences=True, dropout=0.5, recurrent_dropout=0.4)(claims_input)
            encoded_claims = LSTM(512, return_sequences=True)(encoded_claims)
            # encoded_claims = LSTM(128, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)(encoded_claims)
            encoded_claims = Flatten()(encoded_claims)
            concatenate_layers = Dense(512, kernel_regularizer=regularizers.l2(0.001), activation='relu')(encoded_claims)
            concatenate_layers = Dense(256, kernel_regularizer=regularizers.l2(0.001), activation='relu')(encoded_claims)
            concatenate_layers = Dropout(0.6)(concatenate_layers)

            if nb_classes == 3:
                pred_label = Dense(nb_classes, activation='softmax')(concatenate_layers)
            else:
                pred_label = Dense(nb_classes, activation='sigmoid')(concatenate_layers)

            return claims_input, pred_label
    

if __name__ == '__main__':


    training = TrainModel()
    metrics = Metrics()

    max_seq_length = 256

    dataset_name = 'fever_3'
    train_data = pickle.load(open("./bert/datasets/train_"+str(dataset_name)+".pkl", "rb"))

    # claim and sent vector are combined
    # Some embedding files size is > 4Gb (they cannot be pickled in one file)

    # if dataset_name == 'fever_3' or dataset_name == 'fever_binary':

    #     claims_sents_1 = pickle.load(open("./bert/new_embeddings/train_"+str(dataset_name)+"_combinedEmbed_multiling1"+".pkl", "rb"))
    #     claims_sents_2 = pickle.load(open("./bert/new_embeddings/train_"+str(dataset_name)+"_combinedEmbed_multiling2"+".pkl", "rb"))

    #     claims_sents_vec = np.concatenate((claims_sents_1, claims_sents_2))

    # else:
    #     claims_sents_vec = pickle.load(open("./bert/new_embeddings/train_"+str(dataset_name)+"_combinedEmbed_multiling"+".pkl", "rb"))

    if dataset_name == 'fever_sup':
        labels = train_data["lablel"]

    else:
        labels = train_data["label"]

    nb_classes = 3

    if nb_classes == 3:
        labels = np_utils.to_categorical(labels, nb_classes)
        loss = 'categorical_crossentropy'

    else:
        loss = 'binary_crossentropy'

    # embedding_dim = 768
    # claims_input, pred_label = training.lstm_model(max_seq_length, embedding_dim, nb_classes)

    # model = Model([claims_input], pred_label)
    # print (model.summary())

    # early_stopping = EarlyStopping(monitor='val_loss', patience=1)
    # model.compile(optimizer=optimizers.Adam(), loss=loss, metrics=['accuracy'])
    model_path = './fever_simple_claims_model/model_bert_fever_3_acc79.h5'

    # history = model.fit({'claims': claims_sents_vec}, labels, 
    #                             epochs=30, batch_size=128, validation_split=0.1, callbacks=[early_stopping, metrics,
    #                                             ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)])

    test_lstm_model = testModel(dataset_name, model_path, nb_classes)
    test_lstm_model.get_results_on_test_data()