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
from sklearn.metrics import  precision_recall_fscore_support
from keras.preprocessing.sequence import pad_sequences

# this script is used to test the sentence retrieval on complete fever dev dataset
# output of sent ret. model is saved and given as input to claim classification model
# output of claim classification model is fed to the majority voting classifier

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
		self.predicted_evidence = []
		self.true_evidences = []

		with jsonlines.open(data_path, mode='r') as f:
			tmp_dict = {}

			for example in f:

				self.ids.append(example["id"])
				self.claims.append(example["claim"])
				# self.true_evidences.append(example["true_evidence"])
				self.predicted_evidence.append(example["possible_evidence"])
				self.sents.append(example["sentence"])
				# self.labels.append(example["claim_true_label"])
				
				tmp_dict = {'id':self.ids, 'claim':self.claims, 'possible_evidence':self.predicted_evidence, 'sentence':self.sents}

			self.test_data = pd.DataFrame(data=tmp_dict)


if __name__ == '__main__':
	

	parser = argparse.ArgumentParser(description='start training')

	parser.add_argument('task', choices=['sent_retrieval', 'claim_classification'], 
					help="what task should be performed?")

	task = parser.parse_args().task

	#results of sr model are saved in following path
	sr_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/sent_retrieval/fever_blind_binary_lstm.jsonl"


	with open('lstm/trained_tokenizer/tokenizer_claims.pickle', 'rb') as handle:
		claims_tokenizer = pickle.load(handle)

	with open('lstm/trained_tokenizer/tokenizer_sents.pickle', 'rb') as handle:
		sents_tokenizer = pickle.load(handle)


	if task == "sent_retrieval":

		print ("sent retireval using lstm")
		max_claims_length = 50
		max_sents_length = 300
		# this dataset_path contains all the sentences to test sent. ret. model (LSTM)
		dataset_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/fever_blind_sent_ret_dl_models.jsonl"
		model_path = "models_fever_full/sentence_retrieval_models/model_lstm_fever_full_binaryAcc57.h5"
		model = load_model(model_path)

		test_model = testModel(max_claims_length, max_sents_length, model_path, dataset_path)

		test_claims = claims_tokenizer.texts_to_sequences(test_model.test_data["claim"])
		test_sents = sents_tokenizer.texts_to_sequences(test_model.test_data["sentence"])

		test_claims = pad_sequences(test_claims, maxlen=max_claims_length)  #returns array of data
		test_sents = pad_sequences(test_sents, maxlen=max_sents_length)

		print ("test claims ", test_claims.shape)
		print ("test sents ", test_sents.shape)
		
		batch_size = 128
		y_pred = (np.asarray(model.predict({'claims': test_claims, 'sentences': test_sents} , batch_size=batch_size))).round()
		print ("y pred ", len(y_pred))
		print ("saving file and writing data")
	
		with jsonlines.open(sr_results_path, mode="w") as f:

			for j in range(y_pred.shape[0]):
				# 1 shows sentence is related				
				if y_pred[j][0] == 1:

					tmp_dict = {}
					tmp_dict["id"] = int(test_model.test_data["id"].iloc[j])
					# label here is claim label
					# tmp_dict["claim_true_label"] = test_model.test_data["label"].iloc[j]
					tmp_dict["claim"] = test_model.test_data["claim"].iloc[j]
					# tmp_dict["true_evidence"] = test_model.test_data["true_evidence"].iloc[j]
					# tmp_dict["line_num"] = int(test_model.test_data["line_num"].iloc[j])
					tmp_dict["possible_evidence"] = test_model.test_data["possible_evidence"].iloc[j]
					tmp_dict["sentence"] = test_model.test_data["sentence"].iloc[j]		
					f.write(tmp_dict)	
	else:

		print ("inside claim classification lstm")

		max_claims_length = 50
		max_sents_length = 280
		model_path = "models_fever_full/claim_classifier_models/model_lstm_fever_full_binary_claim_classifierAcc559.h5"
		model = load_model(model_path)
		test_model = testModel(max_claims_length, max_sents_length, model_path, sr_results_path)

		cc_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/claim_classifier/fever_blind_binary_lstm.jsonl"

	
		test_claims = claims_tokenizer.texts_to_sequences(test_model.test_data["claim"])
		test_sents = sents_tokenizer.texts_to_sequences(test_model.test_data["sentence"])

		test_claims = pad_sequences(test_claims, maxlen=max_claims_length)  #returns array of data
		test_sents = pad_sequences(test_sents, maxlen=max_sents_length)

		batch_size = 32

		y_pred = (np.asarray(model.predict({'claims': test_claims, 'sentences': test_sents}, 
									batch_size=batch_size))).round()
		
		with jsonlines.open(cc_results_path, mode="w") as f:
			for j in range(y_pred.shape[0]): 
				tmp_dict = {}
				tmp_dict["id"] = int(test_model.test_data["id"].iloc[j])
				# tmp_dict["true_label"] = test_model.test_data["label"].iloc[j] 
				tmp_dict["claim"] = test_model.test_data["claim"].iloc[j] 
				# tmp_dict["true_evidence"] = test_model.test_data["true_evidence"].iloc[j]
				tmp_dict["possible_evidence"] = test_model.test_data["possible_evidence"].iloc[j]
				tmp_dict["sentence"] = test_model.test_data["sentence"].iloc[j]
				if int(y_pred[j][0]) == 1:
					# print ("inside SUPPORTS")
					tmp_dict["predicted_label"] = "SUPPORTS"
				elif int(y_pred[j][0]) == 2:
					# print ("insude REFUTEs")
					tmp_dict["predicted_label"] = "REFUTES"
				else:
					tmp_dict["predicted_label"] = "Not Enough Info"

				f.write(tmp_dict)