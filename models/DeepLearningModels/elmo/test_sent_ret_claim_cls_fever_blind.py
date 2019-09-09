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
		self.predicted_evidence = []

		with jsonlines.open(data_path, mode='r') as f:
			tmp_dict = {}

			for example in f:

				self.ids.append(example["id"])
				self.claims.append(example["claim"])
				# self.true_evidences.append(example["true_evidence"])
				self.predicted_evidence.append(example["possible_evidence"])
				# self.line_num.append(example["line_num"])
				self.sents.append(example["sentence"])

				# if task != "claim_classification":
				# 	self.labels.append(example["label"])
				# 	tmp_dict = {'id':self.ids, 'claim':self.claims,'true_evidence':self.true_evidences, 'line_num':self.line_num,'sentence':self.sents, 'label':self.labels}
				# else:
				# self.labels.append(example["claim_true_label"])
				# tmp_dict = {'id':self.ids, 'claim':self.claims,'true_evidence':self.true_evidences, 'line_num':self.line_num, 'sentence':self.sents, 'label':self.labels}
			tmp_dict = {'id':self.ids, 'claim':self.claims, 'possible_evidence':self.predicted_evidence, 'sentence':self.sents}
			
			self.test_data = pd.DataFrame(data=tmp_dict)


if __name__ == '__main__':
	

	parser = argparse.ArgumentParser(description='start training')

	parser.add_argument('task', choices=['sent_retrieval', 'claim_classification'], 
					help="what task should be performed?")

	# parser.add_argument('elmo_embeddings', choices=['save_embeddings'], 
	# 				help="what task should be performed?")

	# parser.add_argument('flag', choices=['True', 'False'], 
	# 				help="what task should be performed?")

	task = parser.parse_args().task
	# flag = parser.parse_args().flag

	preprocess = preProcessing()

	#results of sr model are saved in following path
	sr_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/sent_retrieval/fever_blind_binary_elmo.jsonl"

	if task == "sent_retrieval":

		max_claims_length = 60
		max_sents_length = 300
		# dataset_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/fever_full_binary_dev_sent_ret.jsonl"
		# dataset_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/fever_full_binary_dev_sent_ret_with_evidences.jsonl"
		model_path = "models_fever_full/sentence_retrieval_models/model_elmo_fever_full_binary_testAcc627.h5"
		model = load_model(model_path)
		dataset_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/fever_blind_sent_ret_dl_models.jsonl"
		test_model = testModel(max_claims_length, max_sents_length, model_path, dataset_path)

		with jsonlines.open(dataset_path, mode='r') as f:
			claims = []
			sents = []
			# labels = []

			for example in f:
			    claims.append(example["claim"])
			    sents.append(example["sentence"])
			    # labels.append(example["label"])

			tmp_dict = {'claim':claims, 'sentence':sents}
			dataframe = pd.DataFrame(data=tmp_dict)


		claims_tokens = []
		sents_tokens = []
		claims_tokens = preprocess.tokenize_data(test_model.test_data, "claim")
		sents_tokens = preprocess.tokenize_data(test_model.test_data, "sentence")
		elmo = ElmoEmbedder(cuda_device=1)
		
		print ("total claim tokens ", len(claims_tokens))
		print ("total sents tokens ", len(sents_tokens))

		batch_size = 32
		total_possible_batch_sizes = len(claims_tokens) / batch_size
		print ("total possible batch sizes ", total_possible_batch_sizes)
		count = 0
		raw_claims = []
		raw_ids = []
		raw_sents = []

		# since padding gives memory error, therefore batches are created
		# 
		with jsonlines.open(sr_results_path, mode="w") as f:
			
			for i in range(len(claims_tokens)):

				if not (i % batch_size):
					
					print ("batch size  ", str(i) + " "+ str(i+batch_size))
					
					claim_embeddings, sent_embeddings, _ = preprocess.create_elmo_embeddings(elmo, claims_tokens[i:i+batch_size], sents_tokens[i:i+batch_size], test_model.test_data, dataset_name="fever_blind_set")
					
					claims_pad, sents_pad = preprocess.to_padding(claim_embeddings, sent_embeddings, None, max_claims_length, max_sents_length)				
					
					# print ("claim pred ", claims_pad.shape)
					# print ("sent pred ", sents_pad.shape)
					
					y_pred = (np.asarray(model.predict({'claims': claims_pad, 'sentences': sents_pad}, 
									batch_size=batch_size))).round()
					# write examples only that has label 1(Related sents)
					

					for j in range(y_pred.shape[0]):
						count += 1
						if y_pred[j][0] == 1:
							tmp_dict = {}

							tmp_dict["id"] = int(test_model.test_data["id"].iloc[i+j])
							# tmp_dict["claim_true_label"] = test_model.test_data["label"].iloc[i+j] 
							tmp_dict["claim"] = test_model.test_data["claim"].iloc[i+j] 
							# tmp_dict["true_evidence"] = test_model.test_data["true_evidence"].iloc[i+j]
							tmp_dict["possible_evidence"] = test_model.test_data["possible_evidence"].iloc[i+j]
							# tmp_dict["line_num"] = int(test_model.test_data["line_num"].iloc[i+j])
							tmp_dict["sentence"] = test_model.test_data["sentence"].iloc[i+j]		
							f.write(tmp_dict)

					count += 1

			# print ("count ", count)
				
	else:


		print ("inside else")
		# print ("Flag ", flag)
		max_claims_length = 50
		max_sents_length = 280
		model_path = "models_fever_full/claim_classifier_models/model_elmo_fever_full_binary_claim_classifierAcc528.h5"
		model = load_model(model_path)
		test_model = testModel(max_claims_length, max_sents_length, model_path, sr_results_path, task)
		cc_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/claim_classifier/fever_blind_binary_elmo.jsonl"


		claims_tokens = []
		sents_tokens = []
		claims_tokens = preprocess.tokenize_data(test_model.test_data, "claim")
		sents_tokens = preprocess.tokenize_data(test_model.test_data, "sentence")
		elmo = ElmoEmbedder(cuda_device=1)
		
		print ("total claim tokens ", len(claims_tokens))
		print ("total sents tokens ", len(sents_tokens))

	
		print ("inside claim classification and not saving elmo embeddings ")

		# claim_embeddings = pickle.load(open("/scratch/kkuma12s/elmo_embeddings/test_claim_elmo_emb_fever_full_claim_clf.pkl", "rb"))
		# sents_embeddings = pickle.load(open("/scratch/kkuma12s/elmo_embeddings/test_sents_elmo_emb_fever_full_claim_clf.pkl", "rb"))
		# print ("clam embeddings ", len(claim_embeddings))
		# print ("sent embeddings ", len(sents_embeddings))

		batch_size = 32
		total_possible_batch_sizes = len(claims_tokens) / batch_size
		print ("total possible batch sizes ", total_possible_batch_sizes)
		count = 0

		# since padding gives memory error, therefore batches are created
		# 
		with jsonlines.open(cc_results_path, mode="w") as f:
			
			for i in range(len(claims_tokens)):

				if not (i % batch_size):

					print ("batch size  ", str(i) + " "+ str(i+batch_size))
					
					claim_embeddings, sent_embeddings, _ = preprocess.create_elmo_embeddings(elmo, claims_tokens[i:i+batch_size], sents_tokens[i:i+batch_size], test_model.test_data, dataset_name="fever_blind_set")
				
					claims_pad, sents_pad = preprocess.to_padding(claim_embeddings, sent_embeddings, None, max_claims_length, max_sents_length)				
									
					y_pred = (np.asarray(model.predict({'claims': claims_pad, 'sentences': sents_pad}, 
									batch_size=batch_size))).round()

					# write examples only that has label 1(Related sents)
					

					for j in range(y_pred.shape[0]):
						count += 1	
						tmp_dict = {}

						tmp_dict["id"] = int(test_model.test_data["id"].iloc[i+j])
						# tmp_dict["true_label"] = test_model.test_data["label"].iloc[i+j] 
						tmp_dict["claim"] = test_model.test_data["claim"].iloc[i+j] 
						# tmp_dict["true_evidence"] = test_model.test_data["true_evidence"].iloc[i+j]
						tmp_dict["possible_evidence"] = test_model.test_data["possible_evidence"].iloc[i+j]
						# tmp_dict["line_num"] = int(test_model.test_data["line_num"].iloc[i+j])
						# tmp_dict["claim"] = test_model.test_data["claim"].iloc[i+j] 
						tmp_dict["sentence"] = test_model.test_data["sentence"].iloc[i+j]
						# tmp_dict["label"] = int(y_pred[j][0])
						if int(y_pred[j][0]) == 1:
							tmp_dict["predicted_label"] = "SUPPORTS"
						elif int(y_pred[j][0]) == 2:
							# print ("insude REFUTEs")
							tmp_dict["predicted_label"] = "REFUTES"
						else:
							tmp_dict["predicted_label"] = "Not Enough Info"
							
						f.write(tmp_dict)

			print ("count ", count)


