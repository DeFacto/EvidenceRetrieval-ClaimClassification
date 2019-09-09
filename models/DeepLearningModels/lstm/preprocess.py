import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import argparse
import pickle

from sklearn.model_selection import train_test_split


class preProcessing():

	'''
	Returns: dictionary of each dataset 
	that has all claims, all labels and sentences
	'''
	def filter_proper_claims_sents(self, datasets):

		return (datasets[datasets['body'].str.len() != 0]) # because some sentences are empty
		

	def combine_train_and_val(self, train_set, val_set, test_set):

		print (test_set)
		result = [train_set, val_set, test_set]
		return pd.concat(result, ignore_index=True)


	def convert_text_to_sequences(self, input_text, vocab_size):

		tokenizer = Tokenizer(num_words=vocab_size)
		tokenizer.fit_on_texts(input_text)
		input_text_index = tokenizer.word_index # return dictionary of wordss {'the':1, 'earth':2, 'is':3}

		# print (input_text)
		
		max_length = max([len(s.split()) for s in input_text])

		print ("max length ", max_length)
		# max length in feversup of claims is 65, and 138 of fever sup sent
		return (tokenizer.texts_to_sequences(input_text), input_text_index)


	def to_padding(self, claims, sentences, labels, vocab_size_of_claims, vocab_size_of_sents, max_claims_length, max_sents_length):

		# print ("inside to padding ")
		train_claims_seq, claim_word_index = self.convert_text_to_sequences(claims, vocab_size_of_claims)
		train_sents_seq, sents_word_index = self.convert_text_to_sequences(sentences, vocab_size_of_sents)

		claims_data = pad_sequences(train_claims_seq, maxlen=max_claims_length)  #returns array of data
		sents_data = pad_sequences(train_sents_seq, maxlen=max_sents_length)
		labels = np.asarray(labels)

		return (claims_data, sents_data, labels, claim_word_index,  sents_word_index)

	# returns dictionary of words
	# key of dictionary is each word and its value is vector of particular dimension
	def parse_glove(self, embedding_dim):


		f = open('../../data/glove/glove.6B.'+str(embedding_dim)+'d.txt')
		embeddings_index = {}
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word]= coefs

		f.close()

		print ("Found word vectors ", len(embeddings_index))

		return embeddings_index

	'''
	Since we are using pre-trained embeddings
	therefore embedding layer expects matrix shape of (max_words, embedding_dim)
	'''
	def create_embedding_matrix(self, max_words, embeddings_index, word_index, embedding_dim):

		# print ("max_words ", max_words)
		embedding_matrix = np.zeros((max_words, embedding_dim))
		# print (" word_index.items() ", word_index.items())
		for word, i in word_index.items():
			if i < max_words:
				embedding_vector = embeddings_index.get(word)
				# words not found in embedding index will be all zeros
				if embedding_vector is not None:
					embedding_matrix[i] = embedding_vector

		return embedding_matrix, embedding_dim


	def load_datafiles(self, dataset_params):
		
		data = dict()
		for p in dataset_params:
			open_data = open(p['EXP_FOLDER'] + p['DATASET'])
			dataframe = pd.read_json(open_data)
			data[str(p['DATASET'][0:-5])] = dataframe # keys are dataset names w/o extension
		            
		return data


	def split_dataset(self, dataset):

		train, test = train_test_split(dataset, test_size=0.1)

		return train, test


if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='start training')

	parser.add_argument('task', choices=["classification-fever", "detection-fever", "google"], help="what task should be performed?")

	task = parser.parse_args().task
	# datasets are already shuffled 
	# train and split
	if task == 'classification-fever':

		# dataset_path = [{'EXP_FOLDER': './data/fever/reject/' , 'DATASET': 'fever_rej.json'}]
		dataset_path = [{'EXP_FOLDER': '../../data/fever/support/' , 'DATASET': 'fever_sup.json'}]

	elif task == 'detection-fever':

		dataset_path = [{'EXP_FOLDER': '../../data/fever/3-class/', 'DATASET': 'fever_3.json'}]

	else:
		dataset_path = [{'EXP_FOLDER': '../../data/google/processed_google_datasets/institution/', 'DATASET': 'institution_process.json'}]

	preprocess = preProcessing()

	dataset = preprocess.load_datafiles(dataset_path)

	# print (len(dataset['fever_rej']))
	print (len(dataset['institution_process']))

	filtered_dataset = preprocess.filter_proper_claims_sents(dataset['institution_process'])
	
	print (len(filtered_dataset))

	train, test = preprocess.split_dataset(filtered_dataset)

	pickle.dump(train, open("./lstm/train_institution.pkl", "wb"))
	pickle.dump(test, open("./lstm/test_institution.pkl", "wb"))
