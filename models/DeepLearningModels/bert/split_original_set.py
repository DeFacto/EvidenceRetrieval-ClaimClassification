import jsonlines

# This script was used to split the sent retrieval dataset into 3 sets
# because of memory error using BERT

def split_and_save_dataset(data_file, results_files):

	data_file = jsonlines.open(data_file, mode="r")
	results_file0 = jsonlines.open(results_files[0], mode="w")
	results_file1 = jsonlines.open(results_files[1], mode="w")
	results_file2 = jsonlines.open(results_files[2], mode="w")
	count = 0

	for example in data_file:

		tmp_dict = {}

		tmp_dict["id"] = example["id"]
		tmp_dict["claim"] = example["claim"]
		tmp_dict["true_evidence"] = example["true_evidence"]
		tmp_dict["claim_true_label"] = example["claim_true_label"]
		# evidence = [example["sentence"], example["line_num"]]
		tmp_dict["line_num"] = example["line_num"]
		tmp_dict["predicted_evidence"] = example["predicted_evidence"]
		tmp_dict["sentence"] = example["sentence"]
		tmp_dict["sent_ret_label"] = example["sent_ret_label"]

		# if evidence in example["true_evidences"]:
		# 	tmp_dict["sent_ret_label"] = 1
		# else:
		# 	tmp_dict["sent_ret_label"] = 0


		
		if count < 60000:
			print ("count ", count)
			results_file0.write(tmp_dict)

		elif count >= 60000 and count < 120000:
			results_file1.write(tmp_dict)

		else:
			results_file2.write(tmp_dict)

		count += 1


if __name__ == '__main__':
	
	data_file = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/fever_full_binary_dev_sent_ret_with_evidences.jsonl"
	results_file1 = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/bert/fever_full_binary_dev_sent_ret_split1.jsonl"
	results_file2 = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/bert/fever_full_binary_dev_sent_ret_split2.jsonl"
	results_file3 = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/bert/fever_full_binary_dev_sent_ret_split3.jsonl"

	split_and_save_dataset(data_file, [results_file1, results_file2, results_file3])
