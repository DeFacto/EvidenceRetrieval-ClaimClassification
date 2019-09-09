import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import numpy as np

import jsonlines
import pandas as pd
from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


intToLabel = {}
intToLabel[0] = "Support"
intToLabel[1] = "Refute"
intToLabel[2] = "Not Enough Info"

class testModel:

	def __init__(self, max_claims_length, max_sents_length, model_path, num_classes, dataset_name=None):

		self.max_claims_length = max_claims_length
		self.max_sents_length = max_sents_length
		self.data_name = dataset_name
		self.model_path = model_path
		self.num_classes = num_classes

		if dataset_name == "fever_full_binary_dev" or dataset_name == "fever_full_binary_dev_claim_labelling":

			if dataset_name == "fever_full_binary_dev":
				test_data = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/"+dataset_name+".jsonl"
			else:
				print ("data set name ", dataset_name)
				test_data = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/claim_classification/"+dataset_name+".jsonl"

			self.claims = []
			self.sents = []
			self.labels = []

			with jsonlines.open(test_data, mode='r') as f:
				tmp_dict = {}
				for example in f:
					self.claims.append(example["claim"])
					self.sents.append(example["sentence"])
					self.labels.append(example["label"])

				tmp_dict = {'claim':self.claims, 'sentence':self.sents, 'label':self.labels}
				self.test_data = pd.DataFrame(data=tmp_dict)

		else:
			self.test_data = pickle.load(open("./elmo/datasets/test_"+str(self.data_name)+".pkl", "rb"))

	def get_results_on_test_data(self, preprocess):

		# embeddings_name = "fever_3"
		embeddings_name = "fever_full_binary_dev"
		# embeddings are also compressed in elmo directory 
		# claim_embeddings = pickle.load(open("/scratch/kkuma12s/elmo_embeddings/test_claim_elmo_emb_"+embeddings_name+".pkl", "rb"))
		# sents_embeddings = pickle.load(open("/scratch/kkuma12s/elmo_embeddings/test_sents_elmo_emb_"+embeddings_name+".pkl", "rb"))
		
		claim_embeddings = pickle.load(open("/scratch/kkuma12s/elmo_embeddings/test_claim_elmo_emb_"+embeddings_name+".pkl", "rb"))
		sents_embeddings = pickle.load(open("/scratch/kkuma12s/elmo_embeddings/test_sents_elmo_emb_"+embeddings_name+".pkl", "rb"))


		if self.data_name == 'fever_sup':
			labels = self.test_data["lablel"]
			
		else:
			labels = self.test_data["label"]

		test_claims_data, test_sents_data= preprocess.to_padding(claim_embeddings, sents_embeddings, labels, self.max_claims_length, self.max_sents_length)
		
		print ("test claims data shape ", test_claims_data.shape)
		model = load_model(self.model_path)
		batch = 64

		if self.num_classes > 1:
			test_labels = np.asarray(labels)
			test_labels = np_utils.to_categorical(test_labels, self.num_classes)
			avg = 'weighted'
		else:
			test_labels = np.asarray(labels)
			avg = 'binary'

		loss, accuracy = model.evaluate({'claims':test_claims_data, 'sentences': test_sents_data} ,
		 									test_labels)
		print ("test loss ", loss)
		print ("test accuracy ", accuracy)

		y_pred = (np.asarray(model.predict({'claims': test_claims_data, 'sentences': test_sents_data} , batch_size=batch))).round()
		# print ("y pred ", y_pred)
		print ("score of lstm ", precision_recall_fscore_support(test_labels, y_pred, average=avg)) 

		with open("elmo_test_results.log", "w") as f:
			f.write("test loss "+str(loss))
			f.write("\r\n test accuracy "+ str(accuracy))
			f.write("\r\n score of lstm "+ str(precision_recall_fscore_support(test_labels, y_pred, average=avg)))


		# convert everything in format for plot confusion matrix
		plot_y_pred = np.argmax(y_pred, axis=1).tolist()
		plot_y_true = np.argmax(test_labels, axis=1).tolist()

		y_pred = [intToLabel[pred] for pred in plot_y_pred]
		t_labels = [intToLabel[t_pred] for t_pred in plot_y_true]
		method="ELMO"
		filename = "elmo_cm.pdf"
		self.save_confusion_matrix(t_labels, y_pred, filename, method)


	def save_confusion_matrix(self, y_true, y_pred, filename, method, ymap=None, figsize=(7,7)):

		# confusion_Matrix = confusion_matrix(y_true, y_pred)
		class_labels = ["Support", "Refute", "Not Enough Info"]
		# class_labels = ["Support", "Refute"]

		if ymap is not None:
			y_pred = [ymap[yi] for yi in y_pred]
			y_true = [ymap[yi] for yi in y_true]
			# labels = [ymap[yi] for yi in class_labels]
		cm = confusion_matrix(y_true, y_pred, labels=class_labels)
		
		cm_sum = np.sum(cm, axis=1, keepdims=True)
		cm_perc = cm / cm_sum.astype(float) * 100
		annot = np.empty_like(cm).astype(str)
		nrows, ncols = cm.shape

		for i in range(nrows):
			for j in range(ncols):
				c = cm[i, j]
				p = cm_perc[i, j]
				if i == j:
				    s = cm_sum[i]
				    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
				elif c == 0:
				    annot[i, j] = ''
				else:
				    annot[i, j] = '%.1f%%\n%d' % (p, c)

		cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
		cm.index.name = 'Actual'
		cm.columns.name = 'Predicted'
		fig, ax = plt.subplots(figsize=figsize)
		sns.heatmap(cm, annot=annot, fmt='', ax=ax, annot_kws={"size": 20})
		plt.title('Confusion Matrix of Multi-class classifier using '+str(method))
		plt.savefig(filename)
