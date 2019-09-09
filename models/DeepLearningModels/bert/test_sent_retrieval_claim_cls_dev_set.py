from keras.models import load_model
# from lstm.preprocess import preProcessing 
from keras.models import load_model
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from keras.utils import np_utils
import pickle
import numpy as np
import gzip
import jsonlines
import pandas as pd

import argparse

# this script is used to save the output of sent ret trained model
# input to trained model is the sentences from fever-full dataset

class testModel:

	def __init__(self, data_path, model_path, count=None):


		self.model_path = model_path

		self.claims = []
		self.sents = []
		self.labels = []
		self.ids = []
		self.line_num = []
		self.true_evidences = []
		self.predicted_evidence = []

		print ("inside test Model constructor")
		count = 0
		with jsonlines.open(data_path, mode='r') as f:
			tmp_dict = {}

			for example in f:
				count += 1
				self.ids.append(example["id"])
				self.claims.append(example["claim"])
				self.true_evidences.append(example["true_evidence"])
				self.predicted_evidence.append(example["predicted_evidence"])
				self.sents.append(example["sentence"])
				self.labels.append(example["claim_true_label"])

			tmp_dict = {'id':self.ids, 'claim':self.claims,'true_evidence':self.true_evidences, 'predicted_evidence':self.predicted_evidence, 'sentence':self.sents, 'label':self.labels}

			self.test_data = pd.DataFrame(data=tmp_dict)


	def load_compressed_pickle_file(self, pickle_file_name):

		with gzip.open(pickle_file_name+'.pgz', 'rb') as f:
			return pickle.load(f)



if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='start training')

	parser.add_argument('task', choices=['sent_retrieval', 'claim_classification'], 
					help="what task should be performed?")

	task = parser.parse_args().task

	#results of sr model are saved in following path
	sr_results_path = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/fever_full_binary_dev_bert_with_evidences.jsonl"

	if task == "sent_retrieval":

		print ("inside sent ret")
		dataset_paths = ["fever_full_binary_dev_sent_ret_split1", "fever_full_binary_dev_sent_ret_split2", "fever_full_binary_dev_sent_ret_split3"]
		model_path = "models_fever_full/sentence_retrieval_models/model_bert_fever_full_binaryAcc73.h5"
		model = load_model(model_path)
		print ("model loaded")
		embeddings_paths = ["fever_full_dev_binary_sent_ret_bert_60k", "fever_full_dev_binary_sent_ret_bert_60k_120k", "fever_full_dev_binary_sent_ret_bert_120k_plus"]
		
		results  = jsonlines.open(sr_results_path, mode="w")

		for i in range(len(dataset_paths)):

			dataset_path = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/bert/"+dataset_paths[i]+".jsonl"
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
					# count += 1
					for j in range(y_pred.shape[0]):
						count += 1
						# print ("j ", j)
						# print ("y_pred outside if", y_pred[j][0]) 
						if y_pred[j][0] == 1:
							tmp_dict = {}
							tmp_dict["id"] = int(test_model.test_data["id"].iloc[i+j])
							tmp_dict["claim_true_label"] = test_model.test_data["label"].iloc[i+j] 
							tmp_dict["claim"] = test_model.test_data["claim"].iloc[i+j] 
							tmp_dict["true_evidence"] = test_model.test_data["true_evidence"].iloc[i+j]
							tmp_dict["predicted_evidence"] = test_model.test_data["predicted_evidence"].iloc[i+j]
							tmp_dict["sentence"] = test_model.test_data["sentence"].iloc[i+j]		
							results.write(tmp_dict)

			print ("count ", count)
			
	else:

		# print ("inside else claim classificatio")
		
		# max_claims_length = 50
		# max_sents_length = 280
		model_path = "models_fever_full/claim_classifier_models/model_bert_fever_full_binary_claim_classifierAcc746.h5"
		model = load_model(model_path)
		
		test_model = testModel(sr_results_path, model_path)
		cc_results_path = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/complete_pipeline/claim_cls/fever_full_dev_bert_with_evidences.jsonl"

		print ("inside claim classification and not saving bert embeddings ")

		# claim_embeddings = pickle.load(open("/scratch/kkuma12s/elmo_embeddings/test_claim_elmo_emb_fever_full_claim_clf.pkl", "rb"))
		claim_embeddings = test_model.load_compressed_pickle_file("/scratch/kkuma12s/new_embeddings/fever_full_dev_claim_cls_bert")
		print ("clam embeddings ", claim_embeddings.shape)
		

		batch_size = 32
		total_possible_batch_sizes = len(claim_embeddings) / batch_size
		print ("total possible batch sizes ", total_possible_batch_sizes)
		count = 0
		raw_claims = []
		raw_ids = []
		raw_sents = []

		# since padding gives memory error, therefore batches are created
		# 
		with jsonlines.open(cc_results_path, mode="w") as f:
			
			for i in range(len(claim_embeddings)):

				if not (i % batch_size):

					print ("batch size  ", str(i) + " "+ str(i+batch_size))
					
					y_pred = (np.asarray(model.predict({'claims': claim_embeddings[i:i+batch_size]}, 
									batch_size=batch_size))).round()
					# write examples only that has label 1(Related sents)
					

					for j in range(y_pred.shape[0]):
						count += 1	
						tmp_dict = {}

						tmp_dict["id"] = int(test_model.test_data["id"].iloc[i+j])
						tmp_dict["true_label"] = test_model.test_data["label"].iloc[i+j] 
						tmp_dict["claim"] = test_model.test_data["claim"].iloc[i+j] 
						tmp_dict["true_evidence"] = test_model.test_data["true_evidence"].iloc[i+j]
						tmp_dict["predicted_evidence"] = test_model.test_data["predicted_evidence"].iloc[i+j]
						tmp_dict["sentence"] = test_model.test_data["sentence"].iloc[i+j]

						if int(y_pred[j][0]) == 1:
							tmp_dict["predicted_label"] = "SUPPORTS"
						elif int(y_pred[j][0]) == 2:
							tmp_dict["predicted_label"] = "REFUTES"
						else:
							tmp_dict["predicted_label"] = "Not Enough Info"


						f.write(tmp_dict)

			print ("count ", count)
	

