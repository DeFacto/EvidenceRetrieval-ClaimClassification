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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from allennlp.commands.elmo import ElmoEmbedder

from elmo.preprocessing import preProcessing

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
				self.labels.append(example["label"])
			tmp_dict = {'id':self.ids, 'claim':self.claims,'true_evidence':self.true_evidences, 
							'line_num':self.line_num, 'sentence':self.sents, 'label':self.labels}

			self.test_data = pd.DataFrame(data=tmp_dict)


if __name__ == '__main__':
	

	parser = argparse.ArgumentParser(description='start training')

	parser.add_argument('task', choices=['sent_retrieval', 'claim_classification'], 
					help="what task should be performed?")

	task = parser.parse_args().task

	preprocess = preProcessing()

	max_claims_length = 60
	max_sents_length = 300
	dataset_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/fever_full_binary_dev_sent_ret.jsonl"
	model_path = "models_fever_full/sentence_retrieval_models/model_elmo_fever_full_binary_testAcc627.h5"
	model = load_model(model_path)
	test_model = testModel(max_claims_length, max_sents_length, model_path, dataset_path)
	
	# claim_embeddings = pickle.load(open("/scratch/kkuma12s/elmo_embeddings/test_claim_elmo_emb_fever_full_binary_sent_ret.pkl", "rb"))
	# sents_embeddings = pickle.load(open("/scratch/kkuma12s/elmo_embeddings/test_sents_elmo_emb_fever_full_binary_sent_ret.pkl", "rb"))
	# print ("clam embeddings ", len(claim_embeddings))
	# print ("sent embeddings ", len(sents_embeddings))

	sr_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_full_dev_sent_ret_results.jsonl"

	batch_size = 32
	total_possible_batch_sizes = len(claim_embeddings) / batch_size
	print ("total possible batch sizes ", total_possible_batch_sizes)
	count = 0
	raw_claims = []
	raw_ids = []
	raw_sents = []

	with jsonlines.open(sr_results_path, mode="w") as f:
		
		for i in range(len(claim_embeddings)):

			if not (i % batch_size):

				print ("batch size  ", str(i) + " "+ str(i+batch_size))
				
				claims_pad, sents_pad, _ = preprocess.to_padding(claim_embeddings[i:i+batch_size], 
								sents_embeddings[i:i+batch_size], None, max_claims_length, max_sents_length)				
				
				y_pred = (np.asarray(model.predict({'claims': claims_pad, 'sentences': sents_pad}, 
								batch_size=batch_size))).round()


				for j in range(y_pred.shape[0]):
				
					
					tmp_dict = {}
					tmp_dict["id"] = int(test_model.test_data["id"].iloc[i+j])					
					tmp_dict["label"] = int(test_model.test_data["label"].iloc[i+j]) 
					tmp_dict["true_evidence"] = test_model.test_data["true_evidence"].iloc[i+j]
					tmp_dict["line_num"] = int(test_model.test_data["line_num"].iloc[i+j])
					tmp_dict["claim"] = test_model.test_data["claim"].iloc[i+j] 
					tmp_dict["sentence"] = test_model.test_data["sentence"].iloc[i+j]
					tmp_dict["predicted_label"] = int(y_pred[j][0])
					f.write(tmp_dict)


	# read sr results path and compute accuracy
	print ("computing accuracy of sent ret. on fever full dev set")
	true_labels = []
	pred_labels = []

	with jsonlines.open(sr_results_path, mode='r') as f:
		tmp_dict = {}

		for example in f:

			
			true_labels.append(example["label"])
			pred_labels.append(example["predicted_label"])

		tmp_dict = {'true_labels':true_labels, 'pred_labels':pred_labels}

		test_data = pd.DataFrame(data=tmp_dict)


	print ("score of elmo " , precision_recall_fscore_support(test_data['true_labels'], test_data['pred_labels'],
												 average='binary'))

	print  ("accuracy score ", accuracy_score(true_labels, pred_labels))
