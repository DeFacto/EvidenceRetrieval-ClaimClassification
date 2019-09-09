from keras.models import load_model
from keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from keras.utils import np_utils
import pickle
import numpy as np
import gzip
import jsonlines
import pandas as pd

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


intToLabel = {}
intToLabel[0] = "Support"
intToLabel[1] = "Refute"
intToLabel[2] = "Not Enough Info"

class testModel:

	def __init__(self, dataset_name, model_path, num_classes):

		self.data_name = dataset_name
		self.model_path = model_path
		self.num_classes = num_classes

		if dataset_name == "fever_full_binary_dev" or dataset_name == "fever_full_binary_dev_claim_labelling" :

			if dataset_name == "fever_full_binary_dev":
				test_data = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/"+dataset_name+".jsonl"
			else:
				test_data = "/scratch/kkuma12s/github/fact-validation/thesis-code/Proof_Extraction/data/fever-full/claim_classification/"+dataset_name+".jsonl"

			# test_data = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/"+dataset_name+".jsonl"
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
			self.test_data = pickle.load(open("./bert/datasets/test_"+str(self.data_name)+".pkl", "rb"))


	def get_results_on_test_data(self):


		if self.data_name == "fever_full_binary_dev" or self.data_name == "fever_full_binary_dev_claim_labelling":
			# claims_sents_vec = pickle.load(open("./bert/new_embeddings/test_"+str(self.data_name)+"_bert_embeddings"+".pkl", "rb"))
			# +str(dataset_name)+"_bert_embeddings"
			with gzip.open("/scratch/kkuma12s/new_embeddings/fever_full_binary_dev_bert_embeddings"+".pgz", 'rb') as f:
				claims_sents_vec = pickle.load(f)

			# claims_sents_vec = pickle.load(open("/scratch/kkuma12s/new_embeddings/"+str(dataset_name)+"_bert_embeddings"+".pgz", "rb"))
		else:

			claims_sents_vec = pickle.load(open("/scratch/kkuma12s/new_embeddings/test_"+str(self.data_name)+"_combinedEmbed_multiling"+".pkl", "rb"))
		
		if self.data_name == 'fever_sup':
			labels = self.test_data["lablel"]
		else:
			labels = self.test_data["label"]

		labels = np.asarray(labels)

		model = load_model(self.model_path)
		batch = 128

		if self.num_classes > 1:
			labels = np_utils.to_categorical(labels, self.num_classes)
			avg = 'weighted'
		else:
			avg = 'binary'

		loss, accuracy = model.evaluate({'claims':claims_sents_vec},
		 									labels)
		print ("test loss ", loss)
		print ("test accuracy ", accuracy)
		y_pred = (np.asarray(model.predict({'claims': claims_sents_vec} , batch_size=batch))).round()
		print ("score of lstm ", precision_recall_fscore_support(labels, y_pred, average=avg))


		with open("bert_test_results.log", "w") as f:
			f.write("test loss "+str(loss))
			f.write("\r\n test accuracy "+ str(accuracy))
			f.write("\r\n score of lstm "+ str(precision_recall_fscore_support(labels, y_pred, average=avg)))

		filename = "bert_cm.pdf"
		method = "LSTM with BERT"

		print ("y pred ", y_pred.shape)
		print ("type yped ", type(y_pred))

		plot_y_pred = np.argmax(y_pred, axis=1).tolist()
		plot_y_true = np.argmax(labels, axis=1).tolist()

		y_pred = [intToLabel[pred] for pred in plot_y_pred]

		t_labels = [intToLabel[t_pred] for t_pred in plot_y_true]

		self.save_confusion_matrix(t_labels, y_pred, filename, method)


	def save_confusion_matrix(self, y_true, y_pred, filename, method, ymap=None, figsize=(8,8)):

		# confusion_Matrix = confusion_matrix(y_true, y_pred)
		class_labels = ["Support", "Refute", "Not Enough Info"]

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
