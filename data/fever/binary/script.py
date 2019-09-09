import pandas as pd 
import pickle 
import numpy as np
import json

from sklearn.model_selection import train_test_split

def split_dataset():

	dataset_path = "train_fever_binary.json"
	dataset = pd.read_json(dataset_path)
	# dataset = json.load(open(dataset_path, "r"))
	
	train, validate = train_test_split(dataset, test_size=0.1)

	with open('train_fever_binary_split.json', 'w') as outfile:

		json.dump(train, outfile)

	with open('validate_fever_binary.json', 'w') as outfile:

		json.dump(validate, outfile)


def create_dataset(dataset):

	
	
	list_ = []
	
	count = 0
	for i in range(len(dataset)):

		dict_ = {}
		dict_["claim"] = dataset["claim"].iloc[i]
		dict_["triples"] = dataset["triples"].iloc[i]
		dict_["sentence"] = dataset["sentence"].iloc[i]
		if dataset["label"].iloc[i] == 0 or dataset["label"].iloc[i] == 1:
			dict_["label"] = 0
		else:
			dict_["label"] = 1
		list_.append(dict_)


	# print (list_)

	with open('fever_binary.json', 'w') as outfile:

		json.dump(list_, outfile)


if __name__ == '__main__':

 
		dataset_path = "../3-class/fever_3.json"
		dataset = pd.read_json(dataset_path)

		# create_dataset(dataset)
		split_dataset()