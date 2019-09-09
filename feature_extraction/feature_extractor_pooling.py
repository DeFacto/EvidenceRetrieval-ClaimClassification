import pandas as pd 
import json
from core.defacto.model import DeFactoModelNLP
from feature_extraction.feature_core import featureCore
import pickle
import argparse
import numpy as np
import multiprocessing as mp


class featureExtractor:

	def __init__(self, task):

		self.task = task
		self.feature_cores = featureCore(self.task)
		self.list_of_defactoNlps = []
	#def extract_features(self, list_of_defactoNlps, model_name)
	def extract_features(self, methods):

		
		if methods == "tfidf":
			return self.feature_cores.get_tf_idf_score(self.list_of_defactoNlps)
		if methods == "vs":
			return self.feature_cores.get_vector_space_score(self.list_of_defactoNlps)

		# if methods == "wmd":
		# 	self.list_of_defactoNlps = self.feature_cores.get_vector_space_score(self.list_of_defactoNlps)			
		# # print ("start tf idf")
		# # list_of_defactoNlps = self.feature_cores.get_tf_idf_score(list_of_defactoNlps)
		# print ("start WMD wmd_score")
		# list_of_defactoNlps = self.feature_cores.get_wmd_score(list_of_defactoNlps)
		# print ("start vector space")
		# list_of_defactoNlps = self.feature_cores.get_vector_space_score(list_of_defactoNlps)
		# print ("saving into file")
		# print ("self.task ", self.task)
		# pickle.dump(list_of_defactoNlps, open((self.task+model_name), 'wb'))

	def pooling(self, file_name):

		# print ("mp.cpu_count() ", mp.cpu_count())
		pool = mp.Pool(processes=3)

		# print(pool.map(self.doubler, data))

		methods = ["tfidf", "vs"]
		# print ("inside pooling")
		temp = []
		for res in pool.imap(self.extract_features, methods):
			# print (res)
			self.list_of_defactoNlps = res
			temp.append(res)
			# print (self.list_of_defactoNlps)
			
		pool.close()
		pool.join()

		print (type(self.list_of_defactoNlps))
		
		for model in temp[1]:
				print (model.method_name)
		print (len(temp))	
		# pickle.dump(self.list_of_defactoNlps, open((file_name+"pooling"), "wb"))

	def doubler(self, number):
		return number * 2

		

	def load_datafiles(self, dataset_params):
		
		data = dict()
		for p in dataset_params:
			open_data = open(p['EXP_FOLDER'] + p['DATASET'])
			dataframe = pd.read_json(open_data)
			data[str(p['DATASET'][0:-5])] = dataframe # keys are dataset names w/o extension
		            
		return data


	'''
	Split ratio 0.6, 0.2, 0.2
	'''
	def split_dataset(self, dataset):

		train, validate, test = np.split(dataset.sample(frac=1), [int(.6*len(dataset)), int(.8*len(dataset))])

		return train, validate, test


	def create_DefactoModel(self, data):

		print ("inside defacto model ")
		# list_of_defactoNlps = []
		for key, value in data.items():
			
			defactoNLPs = []
			# print ("key ", key)
			# train, validate, test = self.split_dataset(data[key])
			# save validation and test dataset 
			
			for i in range(len(value)):

				if key == 'fever_sup':
					if len(value["claim"].iloc[i]) > 0 and len(value["sentence"].iloc[i]) > 0:
						defactoNLPs.append(DeFactoModelNLP(claim=value["claim"].iloc[i], label=value["lablel"].iloc[i], sentences=value["sentence"].iloc[i], extract_triple=False))
				else:

					if len(value["claim"].iloc[i]) > 0 and len(value["sentence"].iloc[i]) > 0:
						defactoNLPs.append(DeFactoModelNLP(claim=value["claim"].iloc[i], label=value["label"].iloc[i], sentences=value["sentence"].iloc[i], extract_triple=False))
					

			self.list_of_defactoNlps = defactoNLPs
			self.pooling("train_file___")
				


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='start training')

	parser.add_argument('task', choices=["classification", "detection"], help="what task should be performed?")

	task = parser.parse_args().task

	if task == 'classification':
		dataset_path = [{'EXP_FOLDER': './data/fever/reject/', 'DATASET': 'train_fever_rej.json'}]

	else:
		dataset_path = [{'EXP_FOLDER': './data/fever/3-class/', 'DATASET': 'fever_3.json'}]

	featureExtractor_ = featureExtractor(task)
	data = featureExtractor_.load_datafiles(dataset_path)
	featureExtractor_.create_DefactoModel(data)

