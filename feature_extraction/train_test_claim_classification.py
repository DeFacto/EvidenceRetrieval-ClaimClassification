import numpy as np
import jsonlines
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from operator import itemgetter 
import pandas as pd
from multiprocessing import Pool
import argparse

from feature_extraction.tfidf import TFIDF
from feature_extraction.vector_space import VectorSpace
from feature_extraction.wmd import wordMoverDistance


'''
# This script is used to evaluate claim classification
It uses correct documents 

'''


class ClaimClassifier:


	def __init__(self):

		self.tfidf = TFIDF()
		self.vs = VectorSpace()
		self.wmd = wordMoverDistance()

	def get_similarity_results(self, data, approach):


		predicted_labels = []
		for index, example in data.iterrows():

			# print (example)
			if approach == "tfidf":	
				_, similarity_score = self.tfidf.apply_tf_idf(example["claims"], example["sentences"])

				#support
				if similarity_score >= 0.5:
					predicted_labels.append(1) 

				#refute
				elif similarity_score >=0.15 and similarity_score <0.5:
					predicted_labels.append(2)
				#NEI
				else:
					predicted_labels.append(0)
					

			# scores.append(similarity_score)
			elif approach == "vs":
					_, similarity_score = self.vs.apply_vector_space(example["claims"], example["sentences"])

					#support
					if similarity_score >= 0.5:
						predicted_labels.append(1)

					#refute
					elif similarity_score >=0.15 and similarity_score <0.5:
						predicted_labels.append(2)
					#NEI
					else:
						predicted_labels.append(0)

			elif approach == "wmd":
					_, similarity_score = self.wmd.compute_wm_distance(example["claims"], example["sentences"])
					# print ("similarity_score ", similarity_score)
					#support
					if similarity_score <= 0.4:
						predicted_labels.append(1)

					#refute
					elif similarity_score >0.4 and similarity_score <=0.9:
						predicted_labels.append(2)
					#NEI
					else:
						predicted_labels.append(0)


			else:
				print ("no approach matched")


		return (data, predicted_labels)


	def compute_results(self, sub_sampled_data, approach, task):


		with jsonlines.open(sub_sampled_data) as f:
			claim_ids = []
			claims = []
			line_nums = []
			sentences = []
			true_evidences = []
			labels = []
			for example in f: 
				claim_ids.append(example["id"])
				claims.append(example["claim"])
				true_evidences.append(example["true_evidence"])
				line_nums.append(example["line_num"])
				sentences.append(example["sentence"])
				labels.append(example["label"])

			tmp_dict = {"claims_ids": claim_ids, "claims": claims, "true_evidences": true_evidences,
								"line_nums": line_nums, "sentences": sentences, "true_label": labels}	

			dataset = pd.DataFrame(tmp_dict)
			
		print (len(dataset))

		if task == 'train':
			train, validate = train_test_split(dataset, test_size=0.1, random_state=42)
			# approach = [approach] * len(train)
			print ("train size ", train.shape)
			print ("validation ", validate.shape)
			true_labels = []
			pred_labels = []

			data, predicted_labels = self.get_similarity_results(train, approach)
			print (" \n ****** Results on Validation data ****** \n ")

			data, predicted_labels = self.get_similarity_results(validate, approach)
			print ("shape of data ", data.shape)
			print ("score of  ", str(approach) + " " + str(precision_recall_fscore_support(validate["true_label"], predicted_labels, average='weighted')))
			print ("accuracy of ", str(approach) + " " + str(accuracy_score(validate["true_label"], predicted_labels)))

		elif task == 'test':

			data, predicted_labels = self.get_similarity_results(dataset, approach)
			print (" \n ****** Results on test data ****** \n ")
			print ("shape of data ", data.shape)
			print ("score of  ", str(approach) + " " + str(precision_recall_fscore_support(data["true_label"], predicted_labels, average='weighted')))
			print ("accuracy of ", str(approach) + " " + str(accuracy_score(data["true_label"], predicted_labels)))

	'''
	preprocess data returns features and labels that are used
	to train classifier
	'''

	def preprocess_data(self, dataset):


		print ("processing the dataset ")

		features = []
		labels = []
		features = pd.DataFrame(columns=['s1','s2','s3','s4','s5'])

		with jsonlines.open(dataset, mode='r') as f:
			for (index, example) in enumerate(f):
				features.loc[index] = example['tf_idf_features']
				if example['true_label'] == 'SUPPORTS':
					labels.append(0)
				elif example['true_label'] == 'REFUTES':
					labels.append(1)
				# NOT ENOUGH INFO
				else:
					labels.append(2)

		labels = np.array(labels)
		print ("feature shape ", len(features))
		print ("labels shape ", labels.shape)

		return (features, labels)
	

	def compute_score(self, true_labels, pred_labels):

		count = 0
		for true_evd_set in true_labels:
			if true_evd_set in pred_labels:
				count += 1
			
		return (count / len(true_labels))

		# print ("score of tfidf ", precision_recall_fscore_support(np.array(true_labels), np.array(pred_labels)))
		# print  ("accuracy score ", accuracy_score(np.array(true_labels), np.array(pred_labels)))


	'''
	evaluate clf 
	'''

	def evaluate_clf(self, x, y_true, model):

		y_pred = model.predict(x)
		print ("score of wmd ", precision_recall_fscore_support(y_true, y_pred, average='weighted'))
		print  ("accuracy score ", accuracy_score(y_true, y_pred))


	def train_clf(self, X, Y, path):

		print ("inside classifier")
		clf = RandomForestClassifier()
		clf.fit(X, Y)

		joblib.dump(clf, path)
		print ("model saved")


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='start training')

	parser.add_argument('task', choices=['train','test'], 
					help="what task should be performed?")

	parser.add_argument('approach', choices=['tfidf', 'vs', 'wmd'], 
					help="Which approach to be used?")

	task = parser.parse_args().task
	approach = parser.parse_args().approach

	print ("task ", task)
	print ("approach ", approach)

	classifier = ClaimClassifier()
	
	# model_path = "./data/fever-full/classifier_results/models/_retrieval_"+str(approach)+"_classifier.pkl"

	if task == 'train':

		# get labels from train_set
		# gold_train = "./data/fever-full/subsample_train_relevant_docs.jsonl"
		#subsample_train_docs_combinedSentences_tfIdfAndDrQA is created from subsample_train_relevant_docs
		# but it contains sentences combined by retrieved docs(using NER and DrQA)
		sub_sampled_train_data = "./data/fever-full/claim_classification/fever_full_binary_train_claim_labelling.jsonl"
		#train data is for classifier
		compute_results = True

		if compute_results:
			#this function takes dataset, compute tf idf sent. sim. and saves result in file
			#first run on sub_sampled_train_data to get features and train RF model
			# then run on test set to get features
			# compute_results(sub_sampled_train_data)
			# features_path = "./data/fever-full/classifier_results/subsample_train_true_docs_"+str(approach)+"_features.jsonl"
			# print ("storing features ")
		
			classifier.compute_results(sub_sampled_train_data, approach, task)
				# print ("overall accuracy ", sum(accuracy)/len(accuracy))
			# classifier.compute_score(true_labels, pred_labels)
			# x, y = classifier.preprocess_data(features_path)
			# x = x.fillna(0)

			
			# classifier.train_clf(x, y, model_path)

	elif task == 'test':
		print ("testing the classifier")
		# here as test set, we use dev set because we have sentence labels
		# gold_test = "./data/fever-full/shared_dev_docs_combinedSentences_tfIdfAndDrQA.jsonl"
		# dev_data = "./data/fever-full/shared_dev_docs_combinedSentences_tfIdfAndDrQA.jsonl"
		dev_data = "./data/fever-full/claim_classification/fever_full_binary_dev_claim_labelling.jsonl"
		compute_results = True
		
		if compute_results:
			#this function takes dataset, compute tf idf sent. sim. and saves result in file
			#first run on sub_sampled_train_data to get features and train RF model
			# then run on test set to get features
			# compute_results(sub_sampled_train_data)
			# features_path = "./data/fever-full/classifier_results/shared_dev_true_docs_"+str(approach)+"_features.jsonl"
			# print ("storing features ")


			classifier.compute_results(dev_data, approach, task)