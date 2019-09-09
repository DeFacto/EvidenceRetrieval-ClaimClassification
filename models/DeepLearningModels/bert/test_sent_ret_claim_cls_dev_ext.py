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

	# parser.add_argument('flag', choices=['train', 'test'], 
	# 				help="what task should be performed?")


	# parser.add_argument('approach', choices=['bert'], 
	# 				help="what task should be performed?")


	parser.add_argument('task', choices=['sent_retrieval', 'claim_classification'], 
					help="what task should be performed?")


	task = parser.parse_args().task
	# flag = parser.parse_args().flag
	# approach = parser.parse_args().approach

	#preprocess = preProcessing()

	#results of sr model are saved in following path
	sr_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/fever_dev_ext_bert.jsonl"

	if task == "sent_retrieval":

		print ("inside sent ret")
		# dataset_paths = ["fever_full_binary_dev_sent_ret_split1", "fever_full_binary_dev_sent_ret_split2", "fever_full_binary_dev_sent_ret_split3"]
		model_path = "models_fever_full/sentence_retrieval_models/model_bert_fever_full_binary_sent_ret_categorical.h5"
		model = load_model(model_path)
		print ("model loaded")
		dataset_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/dev_relevant_docs_sents.jsonl"
		#
		# test_model = testModel(dataset_path, model_path)

		bc = BertClient(check_length=False)

		count = 0
		raw_claims = []
		raw_ids = []
		raw_sents = []
		data_file = jsonlines.open(dataset_path, mode="r")


		with jsonlines.open(sr_results_path, mode="w") as f:

			for example in data_file:

				tmp_dict = {}
				tmp_dict["id"] = example["id"]
				tmp_dict["label"] = example["label"]
				tmp_dict["claim"] =  example["claim"]	
				predicted_sentences = []
				
				sentences = [sent["sentence"] for sent in example["relevant_sentences"]]
				duplicate_claims = [example["claim"] for i in range(len(sentences))]

				sents_pair = [[claim+' ||| '+sent] for claim,sent in zip(duplicate_claims,sentences)]
				i = 0

				for sent in example["relevant_sentences"]:

					claims_sent_vec_combined = bc.encode(sents_pair[i])
					# print ("claims vec ", claims_sent_vec_combined.shape)

					claim = np.array([claims_sent_vec_combined, ])
					# sent =  np.array([sents_pad[i], ])

					y_pred = model.predict({'claims': claims_sent_vec_combined})
					
					# print ("y pred ", y_pred)
					if int(y_pred.argmax(axis=-1)[0]) == 1:

						sent_dict = {}
						sent_dict["id"] = sent["id"]
						sent_dict["line_num"] = sent["line_num"]
						sent_dict["sentence"] = sent["sentence"]
						sent_dict["predicted_label"] = int(y_pred.argmax(axis=-1)[0])
						print ("claim id ", tmp_dict["id"])
						sent_dict["confidences"] = str(y_pred[0].tolist())
						predicted_sentences.append(sent_dict)

					# sent_dict["predicted_label"] = int(y_pred.argmax(axis=-1)[0])


					# print ("predicted_label ", sent_dict["predicted_label"])
					# print ("confidences ", sent_dict["confidences"])
					i += 1


				if len(predicted_sentences) > 0:
					tmp_dict["predicted_sentences"] = predicted_sentences
				# if not relevant evidences then assign final label as NEI
				else:
					sent_dict = {}
					sent_dict["id"] = "null"
					sent_dict["line_num"] = "null"
					sent_dict["sentence"] = "null"
					sent_dict["predicted_label"] = 2

					sent_dict["confidences"] = "null"
					predicted_sentences.append(sent_dict)


					tmp_dict["predicted_sentences"] = predicted_sentences

				f.write(tmp_dict)
				count += 1

				print ("count ", count)
			

	else:

		# print ("inside elese claim classificatio")
		
		# max_claims_length = 50
		# max_sents_length = 280
		model_path = "models_fever_full/claim_classifier_models/model_bert_fever_full_binary_claim_classifierAcc746.h5"
		model = load_model(model_path)
		
		# test_model = testModel(sr_results_path, model_path)
		cc_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/claim_cls/fever_dev_ext_bert.jsonl"

		print ("inside claim classification and not saving bert embeddings ")

		# claim_embeddings = pickle.load(open("/scratch/kkuma12s/elmo_embeddings/test_claim_elmo_emb_fever_full_claim_clf.pkl", "rb"))
		# claim_embeddings = test_model.load_compressed_pickle_file("/scratch/kkuma12s/new_embeddings/fever_full_dev_claim_cls_bert")
		# print ("clam embeddings ", claim_embeddings.shape)
		data_file = jsonlines.open(sr_results_path, mode="r")

		bc = BertClient()

		count = 0

		with jsonlines.open(cc_results_path, mode="w") as f:

			for example in data_file:

				tmp_dict = {}
				tmp_dict["id"] = example["id"]
				tmp_dict["label"] = example["label"]
				tmp_dict["claim"] =  example["claim"]	
				predicted_sentences = []
				
				sentences = [sent["sentence"] for sent in example["predicted_sentences"]]
				duplicate_claims = [example["claim"] for i in range(len(sentences))]

				sents_pair = [[claim+' ||| '+sent] for claim,sent in zip(duplicate_claims,sentences)]
				i = 0

				for sent in example["predicted_sentences"]:

					if sent["sentence"] != "null":

						claims_sent_vec_combined = bc.encode(sents_pair[i])
					# print ("claims vec ", claims_sent_vec_combined.shape)

						claim = np.array([claims_sent_vec_combined, ])
					# sent =  np.array([sents_pad[i], ])
						y_pred = model.predict({'claims': claims_sent_vec_combined})
					
						# print ("y pred ", y_pred)
						if int(y_pred.argmax(axis=-1)[0]) == 0 or int(y_pred.argmax(axis=-1)[0]) == 1:

							sent_dict = {}
							sent_dict["id"] = sent["id"]
							sent_dict["line_num"] = sent["line_num"]
							sent_dict["sentence"] = sent["sentence"]

							if int(y_pred.argmax(axis=-1)[0]) == 0:
								sent_dict["predicted_label"] = "SUPPORTS"
							else:
								sent_dict["predicted_label"] = "REFUTES"

							# print ("claim id ", tmp_dict["id"])
							# sent_dict["confidences"] = str(y_pred[0].tolist())
							predicted_sentences.append(sent_dict)

					i += 1


				if len(predicted_sentences) > 0:
					tmp_dict["predicted_evidence"] = predicted_sentences
				# if not relevant evidences then assign final label as NEI
				else:
					sent_dict = {}
					sent_dict["id"] = "null"
					sent_dict["line_num"] = "null"
					sent_dict["sentence"] = "null"
					sent_dict["predicted_label"] = "Not Enough Info"

					# sent_dict["confidences"] = "null"
					predicted_sentences.append([sent_dict["id"],sent_dict["line_num"]])
					tmp_dict["predicted_evidence"] = predicted_sentences

				f.write(tmp_dict)
				count += 1

				print ("count ", count)
