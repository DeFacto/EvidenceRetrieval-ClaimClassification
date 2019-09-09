import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import numpy as np
import argparse
from random import randint

import jsonlines
import pandas as pd
from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from keras.preprocessing.sequence import pad_sequences

# this script is used to test the sentence retrieval on complete fever dev dataset

# Note dataset used contains unequal number of true +ves and -ves

class testModel:


	def __init__(self, max_claims_length, max_sents_length, model_path, data_path, task=None):


		self.max_claims_length = max_claims_length
		self.max_sents_length = max_sents_length
		self.model_path = model_path

		self.claims = []
		self.sents = []
		self.labels = []
		self.ids = []
		self.line_num = []
		self.true_evidences = []

		with jsonlines.open(data_path, mode='r') as f:
			tmp_dict = {}

			for example in f:

				self.ids.append(example["id"])
				self.claims.append(example["claim"])
				self.true_evidences.append(example["true_evidence"])
				self.line_num.append(example["line_num"])
				self.sents.append(example["sentence"])

				# if task != "claim_classification":
				# 	self.labels.append(example["label"])
				# 	tmp_dict = {'id':self.ids, 'claim':self.claims,'true_evidence':self.true_evidences, 'line_num':self.line_num,'sentence':self.sents, 'label':self.labels}
				# else:
				self.labels.append(example["label"])
				tmp_dict = {'id':self.ids, 'claim':self.claims,'true_evidence':self.true_evidences, 'line_num':self.line_num, 'sentence':self.sents, 'label':self.labels}

			self.test_data = pd.DataFrame(data=tmp_dict)


if __name__ == '__main__':
	

	parser = argparse.ArgumentParser(description='start training')

	parser.add_argument('task', choices=['sent_retrieval', 'claim_classification'], 
					help="what task should be performed?")

	task = parser.parse_args().task

	with open('lstm/trained_tokenizer/tokenizer_claims.pickle', 'rb') as handle:
		claims_tokenizer = pickle.load(handle)

	with open('lstm/trained_tokenizer/tokenizer_sents.pickle', 'rb') as handle:
		sents_tokenizer = pickle.load(handle)


	print ("sent retireval using lstm")
	max_claims_length = 50
	max_sents_length = 300
	# this dataset_path contains all the sentences to test sent. ret. model (LSTM)
	dataset_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/fever_full_binary_dev_sent_ret1111.jsonl"
	model_path = "models_fever_full/sentence_retrieval_models/model_lstm_fever_full_binaryAcc57.h5"
	model = load_model(model_path)

	test_model = testModel(max_claims_length, max_sents_length, model_path, dataset_path)

	test_claims = claims_tokenizer.texts_to_sequences(test_model.test_data["claim"])
	test_sents = sents_tokenizer.texts_to_sequences(test_model.test_data["sentence"])

	test_claims = pad_sequences(test_claims, maxlen=max_claims_length)  #returns array of data
	test_sents = pad_sequences(test_sents, maxlen=max_sents_length)

	print ("test claims ", test_claims.shape)
	print ("test sents ", test_sents.shape)
	
	loss, accuracy = model.evaluate({'claims':test_claims, 'sentences': test_sents} ,
	 									test_model.test_data["label"])
	print ("test loss ", loss)

	print ("test accuracy ", accuracy)


	batch_size = 32
	y_pred = (np.asarray(model.predict({'claims': test_claims, 'sentences': test_sents} , batch_size=batch_size))).round()
	print ("score of lstm ", precision_recall_fscore_support(test_model.test_data["label"], y_pred, average="binary")) 

	f = open("lstm_sent_ret_results.txt", "w")
	f.write("accuracy "+str(accuracy))
	f.write("loss ", +str(loss))
	f.write("precision recall f1  ", +str(precision_recall_fscore_support(test_model.test_data["label"], y_pred, average="binary")))
	f.close()
