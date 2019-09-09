import numpy as np
import jsonlines
import argparse
import random

from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score


from feature_extraction.tfidf import TFIDF
from feature_extraction.vector_space import VectorSpace
from feature_extraction.wmd import wordMoverDistance
from feature_extraction.classifier_rf_fever_full_true_docs import TrainClassifier

'''
Here we combine all three models
This script is not meant to train/finetune tfidf/wmd/vspace thresholds.
These thresholds are learned during individual module training
However, here we just save the results for training also. 
'''

class TestModels:

	def __init__(self, approach, task):

		self.rf_classifier = TrainClassifier()
		self.approach = approach

		if task == "test":
			# self.sr_output = "./data/fever-full/classifier_results/shared_dev_true_docs_"+str(approach)+"_features_new.jsonl"
			# self.sr_output = "./data/fever-full/classifier_results/shared_dev_true_docs_"+str(approach)+"_features_new_k_5.jsonl"
			self.sr_output = "./data/fever-full/classifier_results/shared_dev_true_docs_"+str(approach)+"_features_new_k_5_recall.jsonl"
		else:
			# self.sr_output = "./data/fever-full/classifier_results/subsample_train_true_docs_"+str(approach)+"_features_new.jsonl"
			# self.sr_output = "./data/fever-full/classifier_results/subsample_train_true_docs_"+str(approach)+"_features_new_k_5.jsonl"
			self.sr_output = "./data/fever-full/classifier_results/subsample_train_true_docs_"+str(approach)+"_features_new_k_5_recall.jsonl"

		self.tfidf = TFIDF()
		self.vs = VectorSpace()
		self.wmd = wordMoverDistance()

	# load sentence retrieval results
	# find claim classifier results
	def load_sr_results(self, cc_results_path):

		read_file = jsonlines.open(self.sr_output, mode="r")
		# 
		with jsonlines.open(cc_results_path, mode="w") as f:

			for example in read_file:
				
				tmp_dict = {}
				tmp_dict["id"] = example["id"]

				if example["true_label"] == "SUPPORTS":
					tmp_dict["label"] = "SUPPORTS"
					# tmp_dict["label"] = 1

				elif example["true_label"] == "REFUTES":
					tmp_dict["label"] = "REFUTES"
					# tmp_dict["label"] = 2
				
				else:
					tmp_dict["label"] = "Not Enough Info"
					# tmp_dict["label"] = 0

				tmp_dict["claim"] =  example["claim"]	
				predicted_labels = []

				for sent in example["predicted_sentences"]:

					if self.approach == "tfidf":

						_, similarity_score = self.tfidf.apply_tf_idf(example["claim"], sent)

						#support
						if similarity_score >= 0.5:
							predicted_labels.append("SUPPORTS")
							# predicted_labels.append(1) 

						#refute
						elif similarity_score >=0.15 and similarity_score <0.5:
							predicted_labels.append("REFUTES")
							
							# predicted_labels.append(2)
						#NEI
						else:
							predicted_labels.append("Not Enough Info")
							# predicted_labels.append(0)

					elif self.approach == "vs":

						_, similarity_score = self.vs.apply_vector_space(example["claim"], sent)

						#support
						if similarity_score >= 0.5:
							predicted_labels.append("SUPPORTS")
							# predicted_labels.append(1)

						#refuute
						elif similarity_score >=0.15 and similarity_score <0.5:
							predicted_labels.append("REFUTES")
							# predicted_labels.append(2)
						#NEI
						else:
							predicted_labels.append("Not Enough Info")
							# predicted_labels.append(0)

					elif approach == "wmd":

						_, similarity_score = self.wmd.compute_wm_distance(example["claim"], sent)
						
						# print ("similarity_score ", similarity_score)
						#support
						# 0.4
						if similarity_score <= 0.4:
							predicted_labels.append("SUPPORTS")
							# predicted_labels.append(1)

						#refute
						# similarity_score > 0.4 and similarity_score <=0.9
						elif similarity_score > 0.4 and similarity_score <=0.9:
							predicted_labels.append("REFUTES")
							# predicted_labels.append(2)
						#NEI
						else:
							predicted_labels.append("Not Enough Info")
							# predicted_labels.append(0)

				tmp_dict["claims_labels"] = predicted_labels
				tmp_dict["true_evidence"] = example["true_evidences"]
				tmp_dict["predicted_evidence"] = example["predicted_sentences_ids"]


				f.write(tmp_dict)

	# load claim classification (cc) result
	# cc_results_path is cc results stored
	# final results will path where final label for each label will be stored
	# Save final label based on voting
	def predict_final_label(self, cc_results_path, final_results_path):

		cc_results = jsonlines.open(cc_results_path, mode='r')

		possible_labels = ["Not Enough Info", "SUPPORTS", "REFUTES"]

		with jsonlines.open(final_results_path, mode='w') as f:

			for example in cc_results:
				tmp_dict = {}
				tmp_dict["id"] = example["id"]
				tmp_dict["label"] = example["label"]
				tmp_dict["claim"] = example["claim"]
				# tmp_dict["cc_labels"] = example["claims_labels"]
				tmp_dict["evidence"] = example["true_evidence"]
				tmp_dict["predicted_evidence"] = example["predicted_evidence"]
				count_s_labels = 0
				count_r_labels = 0
				count_nei_labels = 0

				for cc_labels in example["claims_labels"]:
					# if cc_labels == 1:
					if cc_labels == "SUPPORTS":
						count_s_labels += 1
					# elif cc_labels == 2:	
					elif cc_labels == "REFUTES":
						count_r_labels += 1
					else:
						count_nei_labels += 1

				if count_s_labels > count_r_labels:
					tmp_dict["predicted_label"] = "SUPPORTS"
					# tmp_dict["predicted_label"] = 1

				elif count_s_labels == count_r_labels and count_nei_labels == 0:
					random_number = random.randint(1,2)
					tmp_dict["predicted_label"] = possible_labels[random_number]
					# tmp_dict["predicted_label"] = random_number

				elif count_s_labels > 0 and count_r_labels > 0 and count_s_labels == count_r_labels and count_nei_labels > 0:
					random_number = random.randint(1,2)
					tmp_dict["predicted_label"] = possible_labels[random_number]
					# tmp_dict["predicted_label"] = random_number

				elif count_s_labels == 0  and count_r_labels == 0 and count_nei_labels > 0:
					tmp_dict["predicted_label"] = "Not Enough Info"
					# tmp_dict["predicted_label"] = 0

				elif count_r_labels > count_s_labels:
					tmp_dict["predicted_label"] = "REFUTES"
					# tmp_dict["predicted_label"] = 2


				f.write(tmp_dict)


	def compute_score(self, dataset):

		with jsonlines.open(dataset, mode='r') as f:
			
			true_labels = []
			pred_labels = []

			for example in f:

				true_labels.append(example["label"])
				pred_labels.append(example["predicted_label"])

			true_labels = np.array(true_labels)
			pred_labels = np.array(pred_labels)

		print ("score of " + self.approach, precision_recall_fscore_support(true_labels, pred_labels, average='weighted'))
		print  ("accuracy score ", accuracy_score(true_labels, pred_labels))




if __name__ == '__main__':
	

	parser = argparse.ArgumentParser(description="start testing the model")

	parser.add_argument("task", choices=['train','test'])

	parser.add_argument("approach", choices=['tfidf','vs', 'wmd'])

	task = parser.parse_args().task
	approach = parser.parse_args().approach
	print ("task ", task)
	print ("approach ", approach)

	test_classifier = TestModels(approach, task)

	# write cc results based on sr results
	# cc_results_path = "./data/fever-full/complete_sr_cc/fever_full_"+task+"_"+approach+"_sr_cc.jsonl"
	# write complete results 
	# final_results_path = "./data/fever-full/complete_pipeline/fever_full_"+task+"_"+approach+".jsonl"

	
	cc_results_path = "./data/fever-full/complete_sr_cc/fever_full_"+task+"_"+approach+"_sr_cc_k_5_recall_pred_sents.jsonl"
	#cc_results_path = "./data/fever-full/complete_sr_cc/fever_full_"+task+"_"+approach+"_sr_cc_k_5.jsonl"
	final_results_path = "./data/fever-full/complete_pipeline/fever_full_"+task+"_"+approach+"_k_5_recall_pred_sents.jsonl"
	# final_results_path = "./data/fever-full/complete_pipeline/fever_full_"+task+"_"+approach+"_k_5.jsonl"

	find_cc_results = True

	# if claim classification results are not saved
	if find_cc_results:

		test_classifier.load_sr_results(cc_results_path)
		
		# save final predicted label results based on voting
		test_classifier.predict_final_label(cc_results_path, final_results_path)
		test_classifier.compute_score(final_results_path)
		# features, labels = test_classifier.rf_classifier.preprocess_data(final_results_path)
		# test_classifier.rf_classifier.evaluate_clf(features, labels)

	else:
		test_classifier.predict_final_label(cc_results_path, final_results_path)
		print ("==== \n ")
		test_classifier.compute_score(final_results_path)
		# features, labels = test_classifier.rf_classifier.preprocess_data(cc_results_path)
		# test_classifier.rf_classifier.evaluate_clf(features, labels)
		# test_classifier.predict_final_label(cc_results_path, final_results_path)
