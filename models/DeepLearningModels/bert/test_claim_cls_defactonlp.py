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
from bert_serving.client import BertClient

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

				# if count > 100:
				# 	break

				# print ("count ", count)
				count += 1
				self.ids.append(example["id"])
				self.claims.append(example["claim"])
				# self.true_evidences.append(example["true_evidence"])
				self.predicted_evidence.append(example["possible_evidence"])
				self.sents.append(example["sentence"])
				# self.labels.append(example["claim_true_label"])

			tmp_dict = {'id':self.ids, 'claim':self.claims, 'possible_evidence':self.predicted_evidence, 'sentence':self.sents}

			self.test_data = pd.DataFrame(data=tmp_dict)


	def load_compressed_pickle_file(self, pickle_file_name):

		with gzip.open(pickle_file_name+'.pgz', 'rb') as f:
			return pickle.load(f)



if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='start training')

	parser.add_argument('flag', choices=['train', 'test'], 
					help="what task should be performed?")


	parser.add_argument('approach', choices=['lstm', 'elmo', 'bert'], 
					help="what task should be performed?")


	parser.add_argument('task', choices=['sent_retrieval', 'claim_classification'], 
					help="what task should be performed?")


	#this is not used
	# parser.add_argument('bert_embeddings', choices=['save_embeddings'], 
	# 				help="what task should be performed?")

	# parser.add_argument('flag', choices=['True', 'False'], 
	# 				help="what task should be performed?")

	task = parser.parse_args().task
	flag = parser.parse_args().flag
	approach = parser.parse_args().approach

	#preprocess = preProcessing()

	#results of sr model are saved in following path
	sr_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/sent_retrieval/fever_blind_binary_bert.jsonl"

	if task == "sent_retrieval":

		print ("inside sent ret")
		# dataset_paths = ["fever_full_binary_dev_sent_ret_split1", "fever_full_binary_dev_sent_ret_split2", "fever_full_binary_dev_sent_ret_split3"]
		model_path = "models_fever_full/sentence_retrieval_models/model_bert_fever_full_binaryAcc73.h5"
		model = load_model(model_path)
		print ("model loaded")
		dataset_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/fever_blind_sent_ret_dl_models.jsonl"
		test_model = testModel(dataset_path, model_path)

		# embeddings_paths = ["fever_full_dev_binary_sent_ret_bert_60k", "fever_full_dev_binary_sent_ret_bert_60k_120k", "fever_full_dev_binary_sent_ret_bert_120k_plus"]
		
		results  = jsonlines.open(sr_results_path, mode="w")

		bc = BertClient()

		claims = test_model.test_data["claim"].tolist()
		sents = test_model.test_data["sentence"].tolist()

		print ("claims length ", len(claims))

		sents_pair = [[claim+' ||| '+sent] for claim,sent in zip(claims,sents)]

		print ("total sents pairts ", len(sents_pair))


		# for i in range(len(dataset_paths)):

		# 	dataset_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/bert/"+dataset_paths[i]+".jsonl"
		# 	claims_sent_vec_combined = test_model.load_compressed_pickle_file("/scratch/kkuma12s/new_embeddings/"+embeddings_paths[i])
		# 	print ("test data size ", test_model.test_data.shape)			
		# 	print ("inside sentence retrieval and loading bert embeddings ")
		# 	print ("claims vec 1 ", claims_sent_vec_combined.shape)
		batch_size = 32
		total_possible_batch_sizes = len(sents_pair) / batch_size
		print ("total possible batch sizes ", total_possible_batch_sizes)
		count = 1
		# 	raw_claims = []
		# 	raw_ids = []
		# 	raw_sents = []

		# since padding gives memory error, therefore batches are created
		# 
		for i in range(len(sents_pair)):

			# print (" i % batch_size ", i % batch_size)
			if not (i % batch_size):

				# print ("i ", i)
				# print ("i:i+batch_size ", i+batch_size)
				batch_sentences = sents_pair[i:i+batch_size]

				# print ("len batch_sentences ", len(batch_sentences))
				# print ("batch_sentences ", batch_sentences)

				claims_sent_vec_combined = np.empty((batch_size, 768))
				
				count_sent = 0

				for sent in batch_sentences:

					if count_sent == 0:
						# pass
						claims_sent_vec_combined = bc.encode(sent)
						count_sent += 1

					else: 
					    # pass
					    # print ("here")
					    claims_sent_vec_combined = np.vstack((claims_sent_vec_combined, bc.encode(sent)))

				print ("batch size  ", str(i) + " "+ str(i+batch_size))
	
				# print ("claims_sent_vec_combined ", claims_sent_vec_combined.shape)

				# print ("claims_sent_vec_combined[i:i+batch_size] ", claims_sent_vec_combined[i:i+batch_size])

				y_pred = (np.asarray(model.predict({'claims': claims_sent_vec_combined}, 
								batch_size=batch_size))).round()

				
				# count += 1
				for j in range(y_pred.shape[0]):
					count += 1
					# print ("j ", j)
					# print ("y_pred outside if", y_pred[j][0]) 
					if y_pred[j][0] == 1:
						tmp_dict = {}
						# print ("y pred ", y_pred)
						# print ("y_pred inside if ", y_pred[j][0])
						# print ("id ", int(test_model.test_data["id"].iloc[i+j]))
						# print ("i+j ", i+j)
						tmp_dict["id"] = int(test_model.test_data["id"].iloc[i+j])
						# tmp_dict["claim_true_label"] = test_model.test_data["label"].iloc[i+j] 
						tmp_dict["claim"] = test_model.test_data["claim"].iloc[i+j] 
						tmp_dict["possible_evidence"] = test_model.test_data["possible_evidence"].iloc[i+j]
						# tmp_dict["predicted_evidence"] = test_model.test_data["predicted_evidence"].iloc[i+j]
						tmp_dict["sentence"] = test_model.test_data["sentence"].iloc[i+j]		
						results.write(tmp_dict)

			# 	break
			# break


			# print ("count ", count)
			
	else:


		if flag == 'train':
			sent_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/defactonlp/sent_retrieval/predicted_docs_sents_train.jsonl"
			cc_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/defactonlp/claim_classifier/fever_blind_train_"+approach+"_results.jsonl"

		else:
			sent_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/defactonlp/sent_retrieval/predicted_docs_sents_blind.jsonl"
			cc_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/defactonlp/claim_classifier/fever_blind_"+approach+"_results.jsonl"

		# print ("inside elese claim classificatio")
		
		# max_claims_length = 50
		# max_sents_length = 280
		model_path = "models_fever_full/claim_classifier_models/model_bert_fever_full_binary_claim_classifierAcc746.h5"
		model = load_model(model_path)
		
		# test_model = testModel(sr_results_path, model_path)
		# cc_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/claim_classifier/fever_blind_binary_bert.jsonl"

		print ("inside claim classification and not saving bert embeddings ")

		
		bc = BertClient()

		# claims = test_model.test_data["claim"].tolist()
		# sents = test_model.test_data["sentence"].tolist()

		# print ("claims length ", len(claims))

		# sents_pair = [[claim+' ||| '+sent] for claim,sent in zip(claims,sents)]

		# print ("total sents pairts ", len(sents_pair))

		# batch_size = 32
		# total_possible_batch_sizes = len(sents_pair) / batch_size
		# print ("total possible batch sizes ", total_possible_batch_sizes)
		count = 0
		raw_claims = []
		raw_ids = []
		raw_sents = []
		data_file = jsonlines.open(sent_results_path, mode="r")


		with jsonlines.open(cc_results_path, mode="w") as f:

			for example in data_file:



				tmp_dict = {}
				tmp_dict["id"] = example["id"]

				if flag == "train":
					tmp_dict["true_label"] = example["true_label"]

				tmp_dict["claim"] =  example["claim"]	
				predicted_labels = []
				predicted_sentences = []
				
				sentences = [sent["sentence"] for sent in example["predicted_sentences"]]
				duplicate_claims = [example["claim"] for i in range(len(sentences))]

				sents_pair = [[claim+' ||| '+sent] for claim,sent in zip(duplicate_claims,sentences)]

				i = 0

				for sent in example["predicted_sentences"]:

					sent_dict = {}
					sent_dict["id"] = sent["id"]
					sent_dict["line_num"] = sent["line_num"]
					sent_dict["sentence"] = sent["sentence"]

					# if count_sent == 0:
					# 	claims_sent_vec_combined = bc.encode(sent)
					# 	count_sent += 1
					# else:
					# 	claims_sent_vec_combined = np.vstack((claims_sent_vec_combined, bc.encode(sent)))

					# print ("sents_pair[i] ", sents_pair[i])
					claims_sent_vec_combined = bc.encode(sents_pair[i])
					# print ("claims vec ", claims_sent_vec_combined.shape)

					claim = np.array([claims_sent_vec_combined, ])
					# sent =  np.array([sents_pad[i], ])

					y_pred = model.predict({'claims': claims_sent_vec_combined})

					sent_dict["confidences"] = str(y_pred[0].tolist())
					sent_dict["predicted_label"] = int(y_pred.argmax(axis=-1)[0])

					# print ("predicted_label ", sent_dict["predicted_label"])
					# print ("confidences ", sent_dict["confidences"])
					i += 1

					predicted_sentences.append(sent_dict)

				# count += 1

				# if count > 3:
				# 	break
				tmp_dict["predicted_sentences"] = predicted_sentences

				f.write(tmp_dict)
				count += 1

				print ("count ", count)







		# since padding gives memory error, therefore batches are created
		# 
		'''
		with jsonlines.open(cc_results_path, mode="w") as f:
			
			for i in range(len(sents_pair)):

				if not (i % batch_size):



					# print ("i ", i)
					# print ("i:i+batch_size ", i+batch_size)
					batch_sentences = sents_pair[i:i+batch_size]

					# print ("len batch_sentences ", len(batch_sentences))
					# print ("batch_sentences ", batch_sentences)

					claims_sent_vec_combined = np.empty((batch_size, 768))
					
					count_sent = 0

					for sent in batch_sentences:

						if count_sent == 0:
							# pass
							claims_sent_vec_combined = bc.encode(sent)
							count_sent += 1

						else: 
						    # pass
						    # print ("here")
						    claims_sent_vec_combined = np.vstack((claims_sent_vec_combined, bc.encode(sent)))

					print ("batch size  ", str(i) + " "+ str(i+batch_size))


					# print ("batch size  ", str(i) + " "+ str(i+batch_size))
					
					y_pred = (np.asarray(model.predict({'claims': claims_sent_vec_combined}, 
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
						tmp_dict["sentence"] = test_model.test_data["sentence"].iloc[i+j]

						if int(y_pred[j][0]) == 1:
							tmp_dict["predicted_label"] = "SUPPORTS"
						elif int(y_pred[j][0]) == 2:
							tmp_dict["predicted_label"] = "REFUTES"
						else:
							tmp_dict["predicted_label"] = "Not Enough Info"


						f.write(tmp_dict)

			print ("count ", count)
		'''

