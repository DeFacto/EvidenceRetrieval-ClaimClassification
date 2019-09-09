import numpy as np
import jsonlines
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

from operator import itemgetter 
import pandas as pd
from multiprocessing import Pool
import argparse

from feature_extraction.tfidf import TFIDF
from feature_extraction.vector_space import VectorSpace
from feature_extraction.wmd import wordMoverDistance


'''
# This script is used to evaluate sentence retrieval individually
It uses correct documents 

'''
 		 

class SentenceClassifier:


	def __init__(self):

		self.tfidf = TFIDF()
		self.vs = VectorSpace()
		# self.wmd = wordMoverDistance()
		self.count_avg_true_evidences = 0

	def store_tf_idf_results(self, example, approach):


		tmp_dict = {}	
		scores = []
		
		tmp_dict["id"] = example["id"]
		tmp_dict["true_label"] = example["label"]
		tmp_dict["claim"] = example["claim"]
		top_k_sents = 5
		false_negatives_scores = []
		tmp_dict["true_evidences"] = example["true_evidences"]
		tmp_dict["actual_true_positives"] = len(example["true_evidences"])
		tmp_dict["total_sentences"] = len(example["relevant_sentences"])
		tmp_dict["actual_true_negatives"] = len(example["relevant_sentences"]) - len(example["true_evidences"])

		for evidence in example["relevant_sentences"]:

			if approach == "tfidf":	
				_, similarity_score = self.tfidf.apply_tf_idf(example["claim"], evidence["sentence"])

				# print ("similarity_score ", similarity_score)
				# print ("\n")
				if similarity_score > 0.05:
					# print (" scores != 0", similarity_score)
					scores.append(similarity_score)
				else:
					false_negatives_scores.append(similarity_score)

			elif approach == "vs":
				_, similarity_score = self.vs.apply_vector_space(example["claim"], evidence["sentence"])

				# print ("similarity_score ", similarity_score)
				if similarity_score > 0.1:
					scores.append(similarity_score)

				else:
					false_negatives_scores.append(similarity_score)

			else:
				print ("no approach matched")

		#take top k
		sorted_indexes = np.argsort(scores)
		filtered_indexes = []

		if len(scores) == 0:
			# if score is 0, means there is no related sentence
			tmp_dict["predicted_sentences"] = "null"
			tmp_dict["predicted_sentences_ids"] = [ ["null", "null"] ]
			tmp_dict["predicted_sentences"] = [ "null" ]
			# just fill random value
			tmp_dict["tf_idf_features"] = [10000,10000,10000,10000,10000]
			
		elif len(scores) >= top_k_sents:
			# print ("stored indexes ", sorted_indexes[-5:])
			# filtered_indexes = [idx for idx in sorted_indexes[-5:] if scores[idx] > 0.05]

			# print ("filtered_indexes ", filtered_indexes)
			tmp_dict["predicted_sentences"] = itemgetter(*sorted_indexes[-top_k_sents:])(example["relevant_sentences"])
			tmp_dict["predicted_sentences_ids"] = [ [sent["id"], sent["line_num"]] for sent in tmp_dict["predicted_sentences"]]
			tmp_dict["predicted_sentences"] = [ sent["sentence"] for sent in tmp_dict["predicted_sentences"]]
			tmp_dict["tf_idf_features"] = sorted(scores)[-top_k_sents:]
			
		else:
			# if scores size is less than 5, just add extra 10000s to feed to classifier
			# print ("sorted_indexes ", sorted_indexes)
			tmp_dict["predicted_sentences"] = itemgetter(*sorted_indexes)(example["relevant_sentences"])
			# tmp_dict["predicted_sentences"] = example["relevant_sentences"]
			# print ("tmp dict ", tmp_dict["predicted_sentences"])

			if len(sorted_indexes) == 1:
				tmp_dict["predicted_sentences"] = [tmp_dict["predicted_sentences"]]
				# print ("indexes 0")

			tmp_dict["predicted_sentences_ids"] = [ [sent["id"], sent["line_num"]] for sent in tmp_dict["predicted_sentences"]]
			tmp_dict["predicted_sentences"] = [ sent["sentence"] for sent in tmp_dict["predicted_sentences"]]
			tmp_dict["tf_idf_features"] = sorted(scores) + ([10000]*(top_k_sents-len(sorted(scores))))

		
		tmp_dict["accuracy"], t_correct_evds = self.compute_score(tmp_dict["true_evidences"], tmp_dict["predicted_sentences_ids"])
		tmp_dict["predicted_true_positives"] = t_correct_evds
		tmp_dict["predicted_false_positives"] = len(scores) - t_correct_evds
		tmp_dict["predicted_true_negatives"] = len(false_negatives_scores)
		tmp_dict["predicted_false_negatives"] = tmp_dict["predicted_true_negatives"] - len(scores) - t_correct_evds
	
		tmp_dict["Recall"] = self.handle_errors(tmp_dict["predicted_true_positives"], tmp_dict["actual_true_positives"])
		tmp_dict["Precision"] = self.handle_errors(tmp_dict["predicted_true_positives"], tmp_dict["predicted_false_positives"] + tmp_dict["predicted_true_positives"])
		tmp_dict["accuracy_formula"] = self.handle_errors(tmp_dict["predicted_true_positives"], tmp_dict["actual_true_positives"])
		tmp_dict["f1_score"] = self.handle_errors(2 * tmp_dict["Recall"] * tmp_dict["Precision"], tmp_dict["Recall"] + tmp_dict["Precision"])

		return tmp_dict

	# because multiple variables can give zeroDivisionError
	def handle_errors(self, a, b):

		try:
			z = a / b
		except:
			z = 0

		return z


	def store_features(self, sub_sampled_data, features_path, approach):


		sub_sampled_data = jsonlines.open(sub_sampled_data)
		sub_sampled_data = [example for example in sub_sampled_data]

		# for eg in sub_sampled_data:
		# 	self.count_avg_true_evidences = self.count_avg_true_evidences + len(eg["true_evidences"])
		# print ("total evidences ", self.count_avg_true_evidences)
		# print ("leng of true evidence ", self.count_avg_true_evidences/len(sub_sampled_data))

		print ("len of data ", len(sub_sampled_data))
		# features = pd.DataFrame(columns=['S1','S2','S3','S4','S5', 'Label']) # similarity scores of top 5 sents
		approach = [approach] * len(sub_sampled_data)
		print ("approach inside store features ", len(approach))
		pool = Pool(processes=10)

		accuracy = []
		precisions = []
		recalls = []
		f1_scores = []
		formula_acc = []

		count_accurate = 0
		with jsonlines.open(features_path, mode='w') as f:	
			for tmp_dict in pool.starmap(self.store_tf_idf_results, zip(sub_sampled_data, approach)):
				# features.loc[index] = df
				# print ("dictionary ", tmp_dict)
				if tmp_dict["accuracy"] == 1.0:
					count_accurate += 1

				accuracy.append((tmp_dict["accuracy"]))

				precisions.append((tmp_dict["Precision"]))
				recalls.append((tmp_dict["Recall"]))
				f1_scores.append((tmp_dict["f1_score"]))
				formula_acc.append((tmp_dict["accuracy_formula"]))

				f.write(tmp_dict)
		
		pool.close()

		print ("total accurate answers ", count_accurate)

		return accuracy, precisions, recalls, f1_scores, formula_acc



	def store_tf_idf_results_wmd(self, data, f):
        
		count = 0
		accuracies = []
		precisions = []
		recalls = []
		f1_scores = []
		formula_acc = []

		k = 3

		for example in data:
			# print ("count ", count)

			tmp_dict = {}
			scores = []

			false_negatives_scores = []
			tmp_dict["true_evidences"] = example["true_evidences"]
			tmp_dict["actual_true_positives"] = len(example["true_evidences"])
			tmp_dict["total_sentences"] = len(example["relevant_sentences"])
			tmp_dict["actual_true_negatives"] = len(example["relevant_sentences"]) - len(example["true_evidences"])
			
			tmp_dict["id"] = example["id"]
			tmp_dict["true_label"] = example["label"]
			tmp_dict["claim"] = example["claim"]
			tmp_dict["true_evidences"] = example["true_evidences"]

			for evidence in example["relevant_sentences"]:
				_, similarity_score = self.wmd.compute_wm_distance(example["claim"], evidence["sentence"])
				# print (similarity_score)
				if similarity_score != "inf" and similarity_score < 1.5:
					# similarity_score = 4
					scores.append(similarity_score)

				else:
					false_negatives_scores.append(similarity_score)

		    #take top 5
			sorted_indexes = np.argsort(scores)


			if len(scores) == 0:
				tmp_dict["predicted_sentences"] = example["relevant_sentences"]
				tmp_dict["predicted_sentences_ids"] = [ ["null", "null"] ]
				tmp_dict["predicted_sentences"] = [ "null" ]
				# because if similarity is 4, means sentences are not similar
				tmp_dict["tf_idf_features"] = [4,4,4,4,4]

			elif len(scores) >= k:
				tmp_dict["predicted_sentences"] = itemgetter(*sorted_indexes[:k])(example["relevant_sentences"])
				tmp_dict["predicted_sentences_ids"] = [ [sent["id"], sent["line_num"]] for sent in tmp_dict["predicted_sentences"]]
				tmp_dict["predicted_sentences"] = [ sent["sentence"] for sent in tmp_dict["predicted_sentences"]]
				tmp_dict["tf_idf_features"] = sorted(scores)[:k]
    			# df.loc[0] = sorted(scores)[-5:] + [example["label"]]
			else:
				# if scores size is less than 5, just add extra 0s to feed to classifier
				tmp_dict["predicted_sentences"] = itemgetter(*sorted_indexes)(example["relevant_sentences"])

				if len(sorted_indexes) == 1:
					tmp_dict["predicted_sentences"] = [tmp_dict["predicted_sentences"]]

				
				tmp_dict["predicted_sentences_ids"] = [ [sent["id"], sent["line_num"]] for sent in tmp_dict["predicted_sentences"]]
				tmp_dict["predicted_sentences"] = [ sent["sentence"] for sent in tmp_dict["predicted_sentences"]]
				tmp_dict["tf_idf_features"] = sorted(scores) + ([0]*(k-len(sorted(scores))))
				
			tmp_dict["accuracy"], t_correct_evds = self.compute_score(tmp_dict["true_evidences"], tmp_dict["predicted_sentences_ids"])

			tmp_dict["predicted_true_positives"] = t_correct_evds
			tmp_dict["predicted_false_positives"] = len(scores) - t_correct_evds
			tmp_dict["predicted_true_negatives"] = len(false_negatives_scores)
			tmp_dict["predicted_false_negatives"] = tmp_dict["predicted_true_negatives"] - len(scores) - t_correct_evds
		
			tmp_dict["Recall"] = self.handle_errors(tmp_dict["predicted_true_positives"], tmp_dict["actual_true_positives"])
			tmp_dict["Precision"] = self.handle_errors(tmp_dict["predicted_true_positives"], tmp_dict["predicted_false_positives"] + tmp_dict["predicted_true_positives"])
			tmp_dict["accuracy_formula"] = self.handle_errors(tmp_dict["predicted_true_positives"], tmp_dict["actual_true_positives"])
			tmp_dict["f1_score"] = self.handle_errors(2 * tmp_dict["Recall"] * tmp_dict["Precision"], tmp_dict["Recall"] + tmp_dict["Precision"])

			accuracies.append(tmp_dict["accuracy"])
			precisions.append(tmp_dict["Precision"])
			recalls.append(tmp_dict["Recall"])
			f1_scores.append(tmp_dict["f1_score"])
			formula_acc.append(tmp_dict["accuracy_formula"])

			f.write(tmp_dict)
			count += 1

		return accuracies, precisions, recalls, f1_scores, formula_acc


	def store_features_wmd(self, sub_sampled_data, features_path, approach):

		sub_sampled_data = jsonlines.open(sub_sampled_data)
		sub_sampled_data = [example for example in sub_sampled_data]
		print ("len of data ", len(sub_sampled_data))

		with jsonlines.open(features_path, mode='w') as f:
			accuracies , precisions, recalls, f1_scores, formula_acc = self.store_tf_idf_results_wmd(sub_sampled_data, f)

		return accuracies, precisions, recalls, f1_scores, formula_acc




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
	

	# count shows how many evidences were predicted correctly
	def compute_score(self, true_labels, pred_labels):

		count = 0
		for true_evd_set in true_labels:
			if true_evd_set in pred_labels:
				count += 1
			
		return (count / len(true_labels)), count

	'''
	evaluate clf 
	'''

	def evaluate_clf(self, x, y_true, model):

		y_pred = model.predict(x)
		print ("score of tfidf ", precision_recall_fscore_support(y_true, y_pred, average='weighted'))
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

	classifier = SentenceClassifier()
	
	#model_path = "./data/fever-full/classifier_results/models/sentence_retrieval_"+str(approach)+"_classifier.pkl"

	if task == 'train':

		# get labels from train_set
		# gold_train = "./data/fever-full/subsample_train_relevant_docs.jsonl"
		#subsample_train_docs_combinedSentences_tfIdfAndDrQA is created from subsample_train_relevant_docs
		# but it contains sentences combined by retrieved docs(using NER and DrQA)
		sub_sampled_train_data = "./data/fever-full/subsample_train_true_docs_evidences.jsonl"
		#train data is for classifier
		store_features = True

		if store_features:
			#this function takes dataset, compute tf idf sent. sim. and saves result in file
			#first run on sub_sampled_train_data to get features and train RF model
			# then run on test set to get features
			# store_features(sub_sampled_train_data)
			features_path = "./data/fever-full/classifier_results/subsample_train_true_docs_"+str(approach)+"_features_new_k_3_recall.jsonl"
			# features_path = "./data/fever-full/classifier_results/subsample_train_true_docs_"+str(approach)+"_features_new_k_3.jsonl"
			print ("storing features ")
			if approach == "wmd":
				accuracy, precisions, recalls, f1_scores, formula_acc = classifier.store_features_wmd(sub_sampled_train_data, features_path, approach)
				print ("overall accuracy using wmd", sum(accuracy)/len(accuracy))
				print ("overall precision ", sum(precisions)/len(precisions))
				print ("overall recalls ", sum(recalls)/len(recalls))
				print ("overall f1_scores ", sum(f1_scores)/len(f1_scores))
				print ("overall accuformula_accracy ", sum(formula_acc)/len(formula_acc))

				file_name = "feature_extraction/scores_"+str(approach)+"_"+str(task)+"k_3"+".txt"

				f = open(file_name, "w")
				f.write("overall accuracy " + str(sum(accuracy)/len(accuracy)))
				f.write(" \n overall precision "+str(sum(precisions)/len(precisions)))
				f.write("\n overall recalls "+ str(sum(recalls)/len(recalls)))
				f.write("\n overall f1_scores "+ str(sum(f1_scores)/len(f1_scores)))
				f.write("\n overall formula_accracy " + str(sum(formula_acc)/len(formula_acc)))

			else:			
				accuracy, precisions, recalls, f1_scores, formula_acc = classifier.store_features(sub_sampled_train_data, features_path, approach)
				
				file_name = "feature_extraction/scores_"+str(approach)+"_"+str(task)+"k_3"+".txt"
				print ("overall accuracy ", sum(accuracy)/len(accuracy))
				print ("overall precision ", sum(precisions)/len(precisions))
				print ("overall recalls ", sum(recalls)/len(recalls))
				print ("overall f1_scores ", sum(f1_scores)/len(f1_scores))
				print ("overall accuformula_accracy ", sum(formula_acc)/len(formula_acc))

				f = open(file_name, "w")
				f.write("overall accuracy " + str(sum(accuracy)/len(accuracy)))
				f.write(" \n overall precision "+str(sum(precisions)/len(precisions)))
				f.write("\n overall recalls "+ str(sum(recalls)/len(recalls)))
				f.write("\n overall f1_scores "+ str(sum(f1_scores)/len(f1_scores)))
				f.write("\n overall formula_accracy " + str(sum(formula_acc)/len(formula_acc)))

			
	elif task == 'test':
		print ("testing the classifier")
		# here as test set, we euse dev set because we have sentence labels
		# gold_test = "./data/fever-full/shared_dev_docs_combinedSentences_tfIdfAndDrQA.jsonl"
		# dev_data = "./data/fever-full/shared_dev_docs_combinedSentences_tfIdfAndDrQA.jsonl"
		
		# dev_data = "./data/fever-full/shared_dev_true_docs_evidences.jsonl"
		#dev data ext contains sample from nei class
		dev_data_ext = "./data/fever-full/dev_relevant_docs_sents.jsonl"
		store_features = True
		
		if store_features:
			#this function takes dataset, compute tf idf sent. sim. and saves result in file
			#first run on sub_sampled_train_data to get features and train RF model
			# then run on test set to get features
			# store_features(sub_sampled_train_data)
			# features_path = "./data/fever-full/classifier_results/shared_dev_true_docs_"+str(approach)+"_features_new_k_4.jsonl"
			features_path = "./data/fever-full/classifier_results/shared_dev_true_docs_ext"+str(approach)+"_sent_ret.jsonl"
			print ("storing features ")
			
			if approach == "wmd":
				print ("testing wmd")
				accuracy, precisions, recalls, f1_scores, formula_acc = classifier.store_features_wmd(dev_data_ext, features_path, approach)
				# print ("overall accuracy using wmd", sum(accuracy)/len(accuracy))

				file_name = "feature_extraction/scores_"+str(approach)+"_"+str(task)+"k_3"+".txt"
				print ("overall accuracy ", sum(accuracy)/len(accuracy))
				print ("overall precision ", sum(precisions)/len(precisions))
				print ("overall recalls ", sum(recalls)/len(recalls))
				print ("overall f1_scores ", sum(f1_scores)/len(f1_scores))
				print ("overall formula_accracy ", sum(formula_acc)/len(formula_acc))

				f = open(file_name, "w")
				f.write("overall accuracy " + str(sum(accuracy)/len(accuracy)))
				f.write(" \n overall precision "+str(sum(precisions)/len(precisions)))
				f.write("\n overall recalls "+ str(sum(recalls)/len(recalls)))
				f.write("\n overall f1_scores "+ str(sum(f1_scores)/len(f1_scores)))
				f.write("\n overall formula_accracy " + str(sum(formula_acc)/len(formula_acc)))

			else:

				accuracy, precisions, recalls, f1_scores, formula_acc = classifier.store_features(dev_data_ext, features_path, approach)
				# print ("overall accurac", sum(accuracy)/len(accuracy))

				# file_name = "feature_extraction/scores_"+str(approach)+"_"+str(task)+"k_5"+".txt"
				print ("overall accuracy ", sum(accuracy)/len(accuracy))
				print ("overall precision ", sum(precisions)/len(precisions))
				print ("overall recalls ", sum(recalls)/len(recalls))
				print ("overall f1_scores ", sum(f1_scores)/len(f1_scores))
				print ("overall formula_accracy ", sum(formula_acc)/len(formula_acc))

				# f = open(file_name, "w")
				# f.write("overall accuracy " + str(sum(accuracy)/len(accuracy)))
				# f.write(" \n overall precision "+str(sum(precisions)/len(precisions)))
				# f.write("\n overall recalls "+ str(sum(recalls)/len(recalls)))
				# f.write("\n overall f1_scores "+ str(sum(f1_scores)/len(f1_scores)))
				# f.write("\n overall formula_accracy " + str(sum(formula_acc)/len(formula_acc)))
