import pickle
import pandas as pd 
import numpy as np


def load_data():

	train = pickle.load(open("train_fever_rej", "rb"))
	print ("trian data ", len(train))
	validate = pickle.load(open("validate_fever_rej", "rb"))
	print ("validate data ", len(validate))
	test = pickle.load(open("test_fever_rej", "rb"))
	print ("test data ", len(test))

	# print (data)
	# print ("do nogthing")


def split_dataset(dataset):

		train, validate, test = np.split(dataset.sample(frac=1), [int(.6*len(dataset)), int(.8*len(dataset))])

		pickle.dump(train, open("train_fever_rej", "wb"))
		pickle.dump(validate, open("validate_fever_rej", "wb"))
		pickle.dump(test, open("test_fever_rej", "wb"))

		# return train, validate, test

if __name__ == '__main__':
	
	# files_ = ["train_fever_sup", "test_fever_sup", "validate_fever_sup"]

	dataframe = pd.read_json("fever_rej.json")
	split_dataset(dataframe)
	load_data()
	# data = pickle.load(open("train_fever_rej", "rb"))
	# load_data(data)

	# for i in files_:

	# 	data = pickle.load(open("./train_fever_sup", "rb"))
	# 	load_data(data)