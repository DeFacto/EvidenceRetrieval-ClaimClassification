import pickle
import pandas as pd 
import numpy as np
import json

def load_data():

	train = pickle.load(open("train_fever_3", "rb"))
	print ("trian data ", len(train))
	validate = pickle.load(open("validate_fever_3", "rb"))
	print ("validate data ", len(validate))
	test = pickle.load(open("test_fever_3", "rb"))
	print ("test data ", len(test))


def explore_data():

	train = pickle.load(open("test_fever_3", "rb"))
	# print (train)
	list_ = []
	
	count = 0
	for i in range(len(train)):
		# print (train["claim"].iloc[i])
		# print (train["triples"].iloc[i])
		# print (train["sentence"].iloc[i])
		# print (train["label"].iloc[i])
		dict_ = {}
		dict_["claim"] = train["claim"].iloc[i]
		dict_["triples"] = train["triples"].iloc[i]
		dict_["sentence"] = train["sentence"].iloc[i]
		dict_["label"] = str(train["label"].iloc[i])
		list_.append(dict_)


	print (list_)

	with open('test_fever_3.json', 'w') as outfile:

		json.dump(list_, outfile)
			# break
	# 
 #    	json.dump(data, outfile)

if __name__ == '__main__':
	
	explore_data()
	# files_ = ["train_fever_sup", "test_fever_sup", "validate_fever_sup"]
	# convert_df_to_json()
	# dataframe = pd.read_json("fever_3.json", orient='table')
	# print (dataframe)
	# split_dataset()
	# load_data()
