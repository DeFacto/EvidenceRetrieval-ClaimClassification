import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import numpy as np
import argparse
from random import randint

import jsonlines
import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from allennlp.commands.elmo import ElmoEmbedder

from elmo.preprocessing import preProcessing

# this script is used to test the sentence retrieval on complete fever dev dataset
# output of sent ret. model is saved and given as input to claim classification model
# output of claim classification model is fed to the majority voting classifier

# Note dataset used contains unequal number of true +ves and -ves

class testModel:


	def __init__(self):

		pass


		# self.max_claims_length = max_claims_length
		# self.max_sents_length = max_sents_length
		# self.model_path = model_path

		# self.claims = []
		# self.sents = []
		# self.labels = []
		# self.ids = []
		# self.line_num = []
		# self.true_evidences = []
		# self.predicted_evidence = []

		# with jsonlines.open(data_path, mode='r') as f:
		# 	tmp_dict = {}

		# 	for example in f:

		# 		self.ids.append(example["id"])
		# 		self.claims.append(example["claim"])
		# 		# self.true_evidences.append(example["true_evidence"])
		# 		self.predicted_evidence.append(example["possible_evidence"])
		# 		# self.line_num.append(example["line_num"])
		# 		self.sents.append(example["sentence"])

		# 		# if task != "claim_classification":
		# 		# 	self.labels.append(example["label"])
		# 		# 	tmp_dict = {'id':self.ids, 'claim':self.claims,'true_evidence':self.true_evidences, 'line_num':self.line_num,'sentence':self.sents, 'label':self.labels}
		# 		# else:
		# 		# self.labels.append(example["claim_true_label"])
		# 		# tmp_dict = {'id':self.ids, 'claim':self.claims,'true_evidence':self.true_evidences, 'line_num':self.line_num, 'sentence':self.sents, 'label':self.labels}
		# 	tmp_dict = {'id':self.ids, 'claim':self.claims, 'possible_evidence':self.predicted_evidence, 'sentence':self.sents}
			
		# 	self.test_data = pd.DataFrame(data=tmp_dict)



	def tokenize_data(self, data):

	    # tokens_list = [text_to_word_sequence(data[str(token_type)].iloc[i], lower=False) for i in range(len(data))]
		tokens_list = [text_to_word_sequence(data[i],lower=False) for i in range(len(data))]
		return tokens_list



	def create_elmo_embeddings(self, elmo, claims_tokens, sents_tokens, batch_size):


		claim_embeddings = []
		sentence_embeddings = []
		labels = []

		documentIdx = 0
		for elmo_embedding in elmo.embed_sentences(claims_tokens):  
		    
		    # claim_document = documents["claim"].iloc[documentIdx]
		    # Average the 3 layers returned from ELMo
		    avg_elmo_embedding = np.average(elmo_embedding, axis=0)

		    claim_embeddings.append(avg_elmo_embedding)
		    #  since this script is common for all sets
		    # therefore if statement is added as there are no label in fever_full_binary_fev_elmo

		    # if dataset_name != "fever_full_binary_dev_elmo" and dataset_name != "fever_blind_set":        
		    #     labels.append(documents['label'].iloc[documentIdx])

		    documentIdx += 1
		    
		    if documentIdx % 3000 ==0:
		        print ("documents count ", documentIdx)


		documentIdx = 0
		batch_size = 4
		# embed_sentences(tokens, batch_size)
		for elmo_embedding in elmo.embed_sentences(sents_tokens, batch_size):  
		    
		    # if dataset_name == "birth_place" or dataset_name == "institution":
		    #     sent_document = documents["body"].iloc[documentIdx]
		    # else:
		        
		    #     sent_document = documents["sentence"].iloc[documentIdx]


		#                 print ("sentence document ", sent_document)
		#             print (sent_document, "\n")
		#             print (documentIdx)

		    try:

		        # Average the 3 layers returned from ELMo
		        avg_elmo_embedding = np.average(elmo_embedding, axis=0)
		#                 print ("avg elmo embedding ", avg_elmo_embedding.shape)
		    #because some sents have just punc ' (' due to which there is no embeddings
		    except ZeroDivisionError:
		        random_number = randint(4,15)
		        avg_elmo_embedding = np.zeros((random_number, 1024)) 
		        
		    sentence_embeddings.append(avg_elmo_embedding)

		    # Some progress info
		    documentIdx += 1

		    # if documentIdx % 1000 ==0:
		    #     print ("documents count ", documentIdx)
		        
		    
		return claim_embeddings, sentence_embeddings, labels




if __name__ == '__main__':
	

	parser = argparse.ArgumentParser(description='start training')

	parser.add_argument('flag', choices=['train', 'test'], 
					help="what task should be performed?")


	parser.add_argument('approach', choices=['lstm', 'elmo', 'bert'], 
					help="what task should be performed?")


	parser.add_argument('task', choices=['sent_retrieval', 'claim_classification'], 
					help="what task should be performed?")


	# parser.add_argument('elmo_embeddings', choices=['save_embeddings'], 
	# 				help="what task should be performed?")

	# parser.add_argument('flag', choices=['True', 'False'], 
	# 				help="what task should be performed?")

	task = parser.parse_args().task
	flag = parser.parse_args().flag
	approach = parser.parse_args().approach

	preprocess = preProcessing()

	#results of sr model are saved in following path
	sr_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/sent_retrieval/fever_blind_binary_elmo.jsonl"

	if task == "sent_retrieval":
		print ("sent retrieval")

		'''
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
		elmo = ElmoEmbedder(cuda_device=0)
		
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

		'''

	else:


		print ("inside else")
		# print ("Flag ", flag)
		max_claims_length = 50
		max_sents_length = 280
		model_path = "models_fever_full/claim_classifier_models/model_elmo_fever_full_binary_claim_classifierAcc528.h5"
		model = load_model(model_path)

		# test_model = testModel(max_claims_length, max_sents_length, model_path, sr_results_path, task)
		# cc_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/claim_classifier/fever_blind_binary_elmo.jsonl"


		test_model = testModel()
		print ("inside claim classification elmo")

		if flag == 'train':
			sent_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/defactonlp/sent_retrieval/predicted_docs_sents_train.jsonl"
			cc_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/defactonlp/claim_classifier/fever_blind_train_"+approach+"_results.jsonl"

		else:
			sent_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/defactonlp/sent_retrieval/predicted_docs_sents_blind.jsonl"
			cc_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/defactonlp/claim_classifier/fever_blind_"+approach+"_results.jsonl"


		claims_tokens = []
		sents_tokens = []
		
		elmo = ElmoEmbedder(cuda_device=0)
	
		print ("inside claim classification and not saving elmo embeddings ")

		batch_size = 32
		total_possible_batch_sizes = len(claims_tokens) / batch_size
		print ("total possible batch sizes ", total_possible_batch_sizes)
		count = 0

		data_file = jsonlines.open(sent_results_path, mode="r")

		# since padding gives memory error, therefore batches are created
		# 
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

				# print ("total sentences ", len(sentences))
				# print ("total claims ", len(duplicate_claims))

				claims_tokens = test_model.tokenize_data(duplicate_claims)
				sents_tokens = test_model.tokenize_data(sentences)

				# print ("claim tokens ", len(claims_tokens))
				# print ("sent tokens ", len(sents_tokens))

				claim_embeddings, sent_embeddings, _ = test_model.create_elmo_embeddings(elmo, claims_tokens, sents_tokens, batch_size=len(sentences))

				claims_pad, sents_pad = preprocess.to_padding(claim_embeddings, sent_embeddings, None, max_claims_length, max_sents_length)

				# print ("claim pad shape ", claims_pad.shape)

				count += 1
				i = 0
				for sent in example["predicted_sentences"]:

					sent_dict = {}
					sent_dict["id"] = sent["id"]
					sent_dict["line_num"] = sent["line_num"]
					sent_dict["sentence"] = sent["sentence"]

					claim = np.array([claims_pad[i], ])
					sent =  np.array([sents_pad[i], ])
					# print ("claim pad array ", claim.shape)
					y_pred = model.predict({'claims': claim, 'sentences': sent})
					sent_dict["confidences"] = str(y_pred[0].tolist())
					sent_dict["predicted_label"] = int(y_pred.argmax(axis=-1)[0])

					# print ("predicted_label ", sent_dict["predicted_label"])
					i += 1

					predicted_sentences.append(sent_dict)

				tmp_dict["predicted_sentences"] = predicted_sentences

				f.write(tmp_dict)
