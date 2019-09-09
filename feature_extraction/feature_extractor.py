import pandas as pd 
import json
from core.defacto.model import DeFactoModelNLP
from feature_extraction.feature_core import featureCore
import pickle
import argparse
import numpy as np

class featureExtractor:

	def __init__(self, task, use_feature_core=True):

		self.task = task
		
		if use_feature_core:

			self.feature_cores = featureCore(self.task)


	def extract_features(self, list_of_defactoNlps, model_name):

		print ("start tf idf")
		list_of_defactoNlps = self.feature_cores.get_tf_idf_score(list_of_defactoNlps)
		print ("start WMD wmd_score")
		list_of_defactoNlps = self.feature_cores.get_wmd_score(list_of_defactoNlps)
		print ("start vector space")
		list_of_defactoNlps = self.feature_cores.get_vector_space_score(list_of_defactoNlps)
		print ("saving into file")
		# print ("self.task ", self.task)
		pickle.dump(list_of_defactoNlps, open(model_name+".pkl", 'wb'))
    

	def load_datafiles(self, dataset_params):
		
		data = dict()
		for p in dataset_params:
			open_data = open(p['EXP_FOLDER'] + p['split-ds'] + p['DATASET'])
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


		for key, value in data.items():
			
			list_of_defactoNlps = []
			print ("data size " + str(key), len(value))
			for i in range(len(value)):

				if key == 'fever_sup':
					if len(value["claim"].iloc[i]) > 0 and len(value["sentence"].iloc[i]) > 0:
						list_of_defactoNlps.append(DeFactoModelNLP(claim=value["claim"].iloc[i], label=value["lablel"].iloc[i], sentences=value["sentence"].iloc[i], extract_triple=False))
				
				elif key == 'fever_rej' or key == 'fever_3':
					
					if len(value["claim"].iloc[i]) > 0 and len(value["sentence"].iloc[i]) > 0:
						list_of_defactoNlps.append(DeFactoModelNLP(claim=value["claim"].iloc[i], label=value["label"].iloc[i], sentences=value["sentence"].iloc[i], extract_triple=False))

				else:
					pass

			self.extract_features(list_of_defactoNlps, ("train_data_" + key))
				


if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='start training')

	parser.add_argument('task', 
			choices=["bin-classification-fever", "tri-classification-fever", "bin-classification-google",
								"tri-classification-google"], 
					help="what task should be performed?")

	task = parser.parse_args().task

	if task == 'bin-classification-fever':
		dataset_path = [{'EXP_FOLDER': './data/fever/reject/', 'split-ds': 'train_', 
										'DATASET': 'fever_rej.json'}, 
						{'EXP_FOLDER': './data/fever/support/', 'split-ds': 'train_', 
										'DATASET': 'fever_sup.json'},
						{'EXP_FOLDER': './data/fever/binary/', 'split-ds': 'train_', 
										'DATASET': 'fever_binary.json'}]

	elif task == 'tri-classification-fever':
		dataset_path = [{'EXP_FOLDER': './data/fever/3-class/', 'split-ds': 'train_', 
										'DATASET': 'fever_3.json'}]

	elif task == 'bin-classification-google':
		dataset_path = [{'EXP_FOLDER':'./data/google/processed_google_datasets/birth-place/', 
										'split-ds': 'train_', 'DATASET': 'birth-place_process.json'}]

	else:
		dataset_path= [{'EXP_FOLDER':'./data/google/processed_google_datasets/birth-date/', 
											'split-ds': 'train_', 'DATASET': 'birth-date_process.json'},
						{'EXP_FOLDER':'./data/google/processed_google_datasets/death-place/', 
										'split-ds': 'train_', 'DATASET': 'death-place_process.json'},
						{'EXP_FOLDER':'./data/google/processed_google_datasets/education-degree/', 
										'split-ds': 'train_', 'DATASET': 'education-degree_process.json'},
						{'EXP_FOLDER':'./data/google/processed_google_datasets/institution/', 
										'split-ds': 'train_', 'DATASET': 'institution_process.json'}]
										
	featureExtractor_ = featureExtractor(task)
	data = featureExtractor_.load_datafiles(dataset_path)
	featureExtractor_.create_DefactoModel(data)