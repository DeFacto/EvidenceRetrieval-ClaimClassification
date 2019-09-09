import numpy as np
import random

import jsonlines
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class TestPipeline:


	def __init__(self, cc_results_path):

		self.cc_results_path = cc_results_path

		self.ids = []
		self.claims = []
		self.true_evidences = []
		self.predicted_evidence = []
		self.true_label = []
		self.predicted_labels = []

		with jsonlines.open(cc_results_path, mode='r') as f:
			
			tmp_dict = {}
		

			for example in f:

				self.ids.append(example["id"])
				self.true_label.append(example["true_label"])
				self.claims.append(example["claim"])
				self.true_evidences.append(example["true_evidence"])
				self.predicted_evidence.append(example["predicted_evidence"])
				self.predicted_labels.append(example["predicted_label"])
						

				# self.ids.append(example["id"])
				# self.claims.append(example["claim"])
				# # self.true_evidences.append(example["true_evidence"])
				# self.line_num.append(example["line_num"])
				# # self.sents.append(example["sentence"])

				# self.labels.append(example["label"])
			tmp_dict = {'id':self.ids, 'true_label':self.true_label, 'claim':self.claims, 'true_evidence':self.true_evidences, 'predicted_evidence':self.predicted_evidence, 'predicted_label':self.predicted_labels}
			# tmp_dict = {'id':self.ids, 'claim':self.claims, 'line_num':self.line_num,'label':self.labels}

			self.test_data = pd.DataFrame(data=tmp_dict)

	# here we have to combine results of same claim and different evidences
	# and assign them single label
	def combine_results(self, combine_results_path):

		cc_results = jsonlines.open(self.cc_results_path, mode='r')

		unique_ids = self.test_data["id"].unique().tolist()
		print ("unique_ids ", len(unique_ids))

		with jsonlines.open(combine_results_path, mode='w') as f:
			
			old_id = None
			i = 0
			labels = []
			predicted_evidence = []
			count = 0
			# merge duplicates claim and concatenate labels
			for example in cc_results:

				tmp_dict = {}				
				count_duplicate_ids = list(self.test_data.id).count(unique_ids[i]) 
				

				if example["id"] == unique_ids[i]:
					labels.append(example["predicted_label"])
					predicted_evidence.append(example["predicted_evidence"])					
					# print ("claim ",example["claim"])
					# print ("count count_duplicate_ids ", count_duplicate_ids)
					# print ("count ", count)

					if count == count_duplicate_ids-1:
						# if example["true_label"] == "SUPPORTS":
						# 	example["true_label"] = 1
						# elif example["true_label"] == "REFUTES":
						# 	example["true_label"] = 2
						# else:
						# 	example["true_label"] = 0
						tmp_dict = {"id" : example["id"], "true_label":example["true_label"], "claim" : example["claim"], "evidence":example["true_evidence"], "predicted_evidence":predicted_evidence,"multiple_labels" : labels}
						# tmp_dict = {"id" : example["id"], "true_label":example["true_label"], "claim" : example["claim"], "multiple_labels" : labels}
						f.write(tmp_dict)
						i += 1
						labels = []
						predicted_evidence = []
						count = -1

					count += 1

	# This returns all evidences for any label
	def predict_final_label(self, combine_results_path, results_path):


		dataset = jsonlines.open(combine_results_path, mode="r")
		final_labels = ["Not Enough Info", "SUPPORTS", "REFUTES"]

		with jsonlines.open(results_path, mode='w') as f:

			for example in dataset:
				
				tmp_dict = {}
				
				tmp_dict["id"] = example["id"]
				tmp_dict["claim"] = example["claim"]
				tmp_dict["label"] = example["true_label"]
				tmp_dict["evidence"] = example["evidence"]
				tmp_dict["predicted_evidence"] = example["predicted_evidence"]

				count_s_labels = 0
				count_r_labels = 0
				count_nei_labels = 0

				for cc_labels in example["multiple_labels"]:

					if cc_labels == "SUPPORTS":
						count_s_labels += 1

					elif cc_labels == "REFUTES":
						count_r_labels += 1

					else:
						count_nei_labels += 1


				if count_s_labels > count_r_labels:
					tmp_dict["predicted_label"] = "SUPPORTS"
					# tmp_dict["predicted_label"] = 1

				elif count_s_labels == count_r_labels and count_nei_labels == 0:
					random_number = random.randint(1,2)
					tmp_dict["predicted_label"] = final_labels[random_number]
					# tmp_dict["predicted_label"] = random_number

				elif count_s_labels > 0 and count_r_labels > 0 and count_s_labels == count_r_labels and count_nei_labels > 0:
					random_number = random.randint(1,2)
					tmp_dict["predicted_label"] = final_labels[random_number]
					# tmp_dict["predicted_label"] = random_number

				elif count_s_labels == 0  and count_r_labels == 0 and count_nei_labels > 0:
					# tmp_dict["predicted_label"] = 0
					tmp_dict["predicted_label"] = "Not Enough Info"

				elif count_r_labels > count_s_labels:
					tmp_dict["predicted_label"] = "REFUTES"
					# tmp_dict["predicted_label"] = 2


				f.write(tmp_dict)



	# This returns only evidences whose label is predicted
	def predict_final_label_related_evidences(self, combine_results_path, results_path):


		dataset = jsonlines.open(combine_results_path, mode="r")
		final_labels = ["NOT ENOUGH INFO", "SUPPORTS", "REFUTES"]

		with jsonlines.open(results_path, mode='w') as f:

			for example in dataset:
				
				tmp_dict = {}
				
				tmp_dict["id"] = example["id"]
				tmp_dict["claim"] = example["claim"]
				tmp_dict["label"] = example["true_label"]
				tmp_dict["evidence"] = example["evidence"]
				

				count_s_labels = 0
				count_r_labels = 0
				count_nei_labels = 0

				index_count = 0
				support_evidences = []
				refute_evidences = []
				nei_evidences = []

				for cc_labels in example["multiple_labels"]:

					if cc_labels == "SUPPORTS":
						count_s_labels += 1
						support_evidences.append(example["predicted_evidence"][index_count])


					elif cc_labels == "REFUTES":
						count_r_labels += 1
						refute_evidences.append(example["predicted_evidence"][index_count])

					else:
						count_nei_labels += 1
						nei_evidences.append(example["predicted_evidence"][index_count])

					index_count += 1

				if count_s_labels > count_r_labels:
					
					tmp_dict["predicted_label"] = "SUPPORTS"
					tmp_dict["predicted_evidence"] = support_evidences
					# tmp_dict["predicted_label"] = 1

				elif count_s_labels == count_r_labels and count_nei_labels == 0:
					
					tmp_dict["predicted_label"] = "SUPPORTS"
					tmp_dict["predicted_evidence"] = support_evidences

					# random_number = random.randint(1,2)
					# tmp_dict["predicted_label"] = final_labels[random_number]

					# if tmp_dict["predicted_label"] == "SUPPORTS":
					# 	tmp_dict["predicted_evidence"] = support_evidences
					# else:
					# 	tmp_dict["predicted_evidence"] = refute_evidences

					# tmp_dict["predicted_label"] = random_number

				elif count_s_labels > 0 and count_r_labels > 0 and count_s_labels == count_r_labels and count_nei_labels > 0:
					
					tmp_dict["predicted_label"] = "SUPPORTS"
					tmp_dict["predicted_evidence"] = support_evidences
					# random_number = random.randint(1,2)
					# tmp_dict["predicted_label"] = final_labels[random_number]

					# if tmp_dict["predicted_label"] == "SUPPORTS":
					# 	tmp_dict["predicted_evidence"] = support_evidences
					# else:
					# 	tmp_dict["predicted_evidence"] = refute_evidences

					# tmp_dict["predicted_label"] = random_number

				elif count_s_labels == 0  and count_r_labels == 0 and count_nei_labels > 0:
					# tmp_dict["predicted_label"] = 0
					tmp_dict["predicted_label"] = "NOT ENOUGH INFO"
					tmp_dict["predicted_evidence"] = [["null", "null"]]

				elif count_r_labels > count_s_labels:
					
					tmp_dict["predicted_label"] = "REFUTES"
					tmp_dict["predicted_evidence"] = refute_evidences
					# tmp_dict["predicted_label"] = 2
				else:
					print ("condition missing")


				f.write(tmp_dict)



	def compute_score(self, dataset):

		with jsonlines.open(dataset, mode='r') as f:
			
			true_labels = []
			pred_labels = []
			correct_results = 0

			for example in f:

				true_labels.append(example["label"])
				pred_labels.append(example["predicted_label"])

				if example["label"] == example["predicted_label"]:
					correct_results += 1

			print ("total examples ", len(true_labels))

			true_labels = np.array(true_labels)
			pred_labels = np.array(pred_labels)


		file_ = open("bert_complete_results.txt", mode="w")
		file_.write("precision_recall_fscore_support "+ str(precision_recall_fscore_support(true_labels, pred_labels, average='weighted')))
		file_.write(" \n accuracy score " + str(accuracy_score(true_labels, pred_labels)))
		file_.write("\n")
		print ("score of bert ",  precision_recall_fscore_support(true_labels, pred_labels, average='weighted'))
		print  ("accuracy score ", accuracy_score(true_labels, pred_labels))
		print ("correct results ", correct_results)
		print ("manual accuracy ", correct_results/len(true_labels))



if __name__ == '__main__':
	
	cc_results_path = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/complete_pipeline/claim_cls/fever_full_dev_bert_with_evidences.jsonl"
	# combine result path (combine ssimilar ids/claim and concatenates labels)
	combine_results_path = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/complete_pipeline/claim_cls/fever_full_test_bert_combined_results_with_evidences.jsonl"
	
	testPipeline = TestPipeline(cc_results_path)
	testPipeline.combine_results(combine_results_path)

	# final_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/fever_full_test_bert_with_evidences.jsonl"
	print ("complete combine result function")
	final_results_path = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/complete_pipeline/fever_full_test_bert_label_related_evidences.jsonl"
	# testPipeline.predict_final_label(combine_results_path, final_results_path)

	testPipeline.predict_final_label_related_evidences(combine_results_path, final_results_path)
	testPipeline.compute_score(final_results_path)