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
				self.true_label.append(example["label"])
				self.claims.append(example["claim"])
				self.true_evidences.append(example["true_evidences"])
				# self.predicted_evidence.append(example["possible_evidence"])
				# self.predicted_labels.append(example["predicted_label"])

			tmp_dict = {'id':self.ids,  'claim':self.claims,  'label':self.true_label, 'true_evidence':self.true_evidences}
			# tmp_dict = {'id':self.ids, 'claim':self.claims, 'line_num':self.line_num,'label':self.labels}

			self.test_data = pd.DataFrame(data=tmp_dict)

	#
	def predict_final_label(self, combine_results_path, results_path):


		dataset = jsonlines.open(combine_results_path, mode="r")
		final_labels = ["NOT ENOUGH INFO", "SUPPORTS", "REFUTES"]

		i = 0
		original_set = self.test_data
		with jsonlines.open(results_path, mode='w') as f:

			for example in dataset:
				
				tmp_dict = {}
				
				tmp_dict["id"] = example["id"]
				tmp_dict["label"] = example["label"]
				tmp_dict["claim"] = example["claim"]
				tmp_dict["true_evidence"] = original_set["true_evidence"].iloc[i]
				# tmp_dict["label"] = example["true_label"]
				# tmp_dict["evidence"] = example["evidence"]
				#tmp_dict["predicted_evidence"] = example["predicted_evidence"]
				i += 1
				count_s_labels = 0
				count_r_labels = 0
				count_nei_labels = 0
				evidences = []

				# print ("len of predicted_evidence ", len(example["predicted_evidence"]))
				for evidence in example["predicted_evidence"]:

					if "id" in evidence:
					# if evidence["id"]:
					# 	evidences.append([evidence["id"],evidence["line_num"]])
					# else:
					# 	evidences.append([evidence])
						# print (type(int(evidence["line_num"])))
						if [evidence["id"], evidence["line_num"]] not in evidences:

							evidences.append([evidence["id"], evidence["line_num"]])

						
						if evidence["predicted_label"] == "SUPPORTS":
							count_s_labels += 1
							

						elif evidence["predicted_label"] == "REFUTES":
							count_r_labels += 1

						else:
							count_nei_labels += 1

						tmp_dict["predicted_evidence"] = evidences


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
							tmp_dict["predicted_label"] = "NOT ENOUGH INFO"

						elif count_r_labels > count_s_labels:
							tmp_dict["predicted_label"] = "REFUTES"


					else:
						tmp_dict["predicted_evidence"] = [evidence]
						tmp_dict["predicted_label"] = "NOT ENOUGH INFO"
	
					# tmp_dict["predicted_label"] = 2


				f.write(tmp_dict)


if __name__ == '__main__':
	
	# original dataset has true evidence field
	original_set = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/dev_relevant_docs_sents.jsonl"

	testPipeline = TestPipeline(original_set)
	
	# in cc dataset, we forgot to add true evidence so we will add here
	
	cc_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/claim_cls/fever_dev_ext_bert.jsonl"
	

	#testPipeline.combine_results(combine_results_path)

	final_results_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/fever_full_dev_ext_bert_with_evidences.jsonl"
	print ("complete combine result function")
	testPipeline.predict_final_label(cc_results_path, final_results_path)
