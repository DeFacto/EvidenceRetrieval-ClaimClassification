from keras.models import load_model
# from lstm.preprocess import preProcessing 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from keras.utils import np_utils
import pickle
import numpy as np
import gzip
import jsonlines
import pandas as pd

import argparse

#This scripts tests the trained sentence retireval module

class testModel:

	def __init__(self, data_path, model_path, count=None):


		self.model_path = model_path
		self.claims = []
		self.sents = []
		self.labels = []
		self.ids = []
		self.line_num = []
		self.true_evidences = []
		
		print ("inside test Model constructor")
		count = 0
		with jsonlines.open(data_path, mode='r') as f:
			tmp_dict = {}

			for example in f:

				# if count > 100:
				# 	break

				# print ("count ", count)
				count += 1
				self.ids.append(example["id"])
				self.claims.append(example["claim"])
				self.true_evidences.append(example["true_evidence"])
				self.line_num.append(example["line_num"])
				self.sents.append(example["sentence"])
				self.labels.append(example["label"])
			tmp_dict = {'id':self.ids, 'claim':self.claims,'true_evidence':self.true_evidences, 'line_num':self.line_num, 'sentence':self.sents, 'label':self.labels}

			self.test_data = pd.DataFrame(data=tmp_dict)

	def load_compressed_pickle_file(self, pickle_file_name):

		with gzip.open(pickle_file_name+'.pgz', 'rb') as f:
			return pickle.load(f)


if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='start training')

	parser.add_argument('task', choices=['sent_retrieval', 'claim_classification'], 
					help="what task should be performed?")

	#this is not used
	#preprocess = preProcessing()

	#results of sr model are saved in following path
	sr_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/fever_full_binary_complete_dev_bert.jsonl"

	print ("inside sent ret")
	dataset_paths = ["fever_full_binary_dev_sent_ret_split1_new", "fever_full_binary_dev_sent_ret_split2_new", "fever_full_binary_dev_sent_ret_split3_new"]
	model_path = "models_fever_full/sentence_retrieval_models/model_bert_fever_full_binaryAcc73.h5"
	model = load_model(model_path)
	print ("model loaded")
	embeddings_paths = ["fever_full_dev_binary_sent_ret_bert_60k", "fever_full_dev_binary_sent_ret_bert_60k_120k", "fever_full_dev_binary_sent_ret_bert_120k_plus"]
	
	results  = jsonlines.open(sr_results_path, mode="w")

	for i in range(len(dataset_paths)):

		dataset_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/bert/"+dataset_paths[i]+".jsonl"
		test_model = testModel(dataset_path, model_path)
		claims_sent_vec_combined = test_model.load_compressed_pickle_file("/scratch/kkuma12s/new_embeddings/"+embeddings_paths[i])
		print ("test data size ", test_model.test_data.shape)			
		print ("inside sentence retrieval and loading bert embeddings ")
		print ("claims vec 1 ", claims_sent_vec_combined.shape)
		batch_size = 32
		total_possible_batch_sizes = len(claims_sent_vec_combined) / batch_size
		print ("total possible batch sizes ", total_possible_batch_sizes)
		count = 1
		raw_claims = []
		raw_ids = []
		raw_sents = []

	# since padding gives memory error, therefore batches are created
	# 
		
		for i in range(claims_sent_vec_combined.shape[0]):

			if not (i % batch_size):

				print ("batch size  ", str(i) + " "+ str(i+batch_size))
				
				
				y_pred = (np.asarray(model.predict({'claims': claims_sent_vec_combined[i:i+batch_size]}, 
								batch_size=batch_size))).round()
				
				for j in range(y_pred.shape[0]):
					count += 1
					tmp_dict = {}
					tmp_dict["id"] = int(test_model.test_data["id"].iloc[i+j])
					tmp_dict["label"] = int(test_model.test_data["label"].iloc[i+j]) 
					tmp_dict["true_evidence"] = test_model.test_data["true_evidence"].iloc[i+j]
					tmp_dict["line_num"] = int(test_model.test_data["line_num"].iloc[i+j])
					tmp_dict["claim"] = test_model.test_data["claim"].iloc[i+j] 
					tmp_dict["sentence"] = test_model.test_data["sentence"].iloc[i+j]	
					tmp_dict["predicted_label"] = int(y_pred[j][0])	
					results.write(tmp_dict)

			print ("count ", count)


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


	print ("score of bert " , precision_recall_fscore_support(test_data['true_labels'], test_data['pred_labels'],
												 average='binary'))

	print  ("accuracy score ", accuracy_score(true_labels, pred_labels))
