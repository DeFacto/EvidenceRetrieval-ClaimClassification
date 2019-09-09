import jsonlines 

'''
This script takes the final output of majority vote classifier and add the claims that were 
ignored during sentence retrieval
'''

def read_and_create_set(original_set, predicted_claims_set, final_output, dataset_name):

	original_set = jsonlines.open(original_set, mode="r")
	
	
	with jsonlines.open(final_output, mode="w") as f:

		count = 0
		not_count = 0
		for example in original_set:
				
			tmp_dict = {}
			id_found = False
			file_set = jsonlines.open(predicted_claims_set, mode="r")
			# print ("example id ", example["id"])
			for claim in file_set:
				
				if dataset_name == "fever_dev":

					if example["id"] == claim["id"]:
						count = count + 1
						# print ("ccount ", count)
						f.write(claim)
						id_found = True

				else:
					if example["id"] == claim["id"]:
						count += 1
						tmp_dict["id"] = claim["id"]
						tmp_dict["predicted_label"] = claim["predicted_label"]
						tmp_dict["predicted_evidence"] = claim["predicted_evidence"]
						# if tmp_dict["predicted_label"] == "Not Enough Info":
						# 	tmp_dict["predicted_evidence"] = []
						# else:
						# 	tmp_dict["predicted_evidence"] = claim["predicted_evidence"] 
						
						f.write(claim)
						id_found = True

					
			if not id_found:

				not_count += 1
				# print ("not count ", not_count)
				if dataset_name == "fever_dev":

				
					tmp_dict["id"] = example["id"]
					tmp_dict["label"] = example["label"]
					tmp_dict["claim"] = example["claim"]
					tmp_dict["evidence"] = example["true_evidences"]
					tmp_dict["predicted_evidence"] = [["null", "null"]]
					tmp_dict["predicted_label"] = "NOT ENOUGH INFO"
					f.write(tmp_dict)

				else:

					tmp_dict["id"] = example["id"]
					tmp_dict["predicted_label"] = "NOT ENOUGH INFO"
					tmp_dict["predicted_evidence"] = []
					
					f.write(tmp_dict)

					
		print ("count ", count)
		print ("not count ", not_count)




if __name__ == '__main__':
	

	# For fever dev
	dataset_name = "fever_dev"
	original_test_set = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/shared_dev_true_docs_evidences.jsonl"
	predicted_claims_set = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/complete_pipeline/fever_full_test_bert_label_related_evidences.jsonl"

	# support (support was preferred)
	final_output = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/complete_pipeline/fever_full_test_bert_related_evidences_final_output_support.jsonl"
	# final_output = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/fever_full_test_bert_related_evidences_final_output_refute.jsonl"
	read_and_create_set(original_test_set, predicted_claims_set, final_output, dataset_name)

	# For Fever blind
	# dataset_name = "fever_blind_set"
	# original_blind_set = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/shared_task_test.jsonl"
	# predicted_claims_set = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/claim_classifier/fever_blind_bert_unique_ids.jsonl"
	# final_output = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/final_output/predictions_bert.jsonl"
	# read_and_create_set(original_blind_set, predicted_claims_set, final_output, dataset_name)