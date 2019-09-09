import pickle
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from core.defacto.model import DeFactoModelNLP
import matplotlib.pyplot as plt

from feature_extraction.feature_extractor import featureExtractor
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
#dependencies are installed in tf environment 
class tuneModel:


	def __init__(self, task=None):
		
		self.featureExtractor = featureExtractor(task)


	def test_on_validation_set(self, validation_set, task):

		for key, value in validation_set.items():
			list_of_defactoNlps = []
			
			for i in range(len(value)):
						list_of_defactoNlps.append(DeFactoModelNLP(claim=value["claim"].iloc[i], label=value["label"].iloc[i], sentences=value["sentence"].iloc[i], extract_triple=False))

			self.featureExtractor.extract_features(list_of_defactoNlps, ("./fever_pickle_models/test_bin_fever_binary"))


	def save_confusion_matrix(self, y_true, y_pred, filename, method, ymap=None, figsize=(7,7)):

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

		# for cm in confusion_Matrix:
		# 	fig = plt.figure()
		# 	plt.matshow(confusion_Matrix)
		# 	plt.title('Confusion Matrix of Multi-class classifier using '+str(method))
		# 	plt.colorbar()
		# 	plt.ylabel('True Label')
		# 	plt.xlabel('Predicated Label')
		# 	fig.savefig('confusion_matrix_'+str(method)+'.jpg')



	def compute_score(self, list_of_defactoNlps, task):

		y_pred_tfidf = []
		y_pred_wmd = []
		y_pred_vspace = []
		y_true = []
		print ("inside compute score")
		for model in list_of_defactoNlps:
				
			# print ("predicted label")
			# print (model.method_name['vspace']['Detection']['pred_label'])
			# print (model.method_name['tfidf'][task]['pred_label'])
			# print (model.method_name['tfidf'])
			tfidf = model.method_name['tfidf'][task]['pred_label']
			vspace = model.method_name['vspace'][task]['pred_label']
			wmd = model.method_name['wmd'][task]['pred_label']

			if  tfidf == 0:
				y_pred_tfidf.append("Support")

			elif tfidf == 1:
				y_pred_tfidf.append("Refute")

			else:
				y_pred_tfidf.append("Not Enough Info")

			if vspace == 0:
				y_pred_vspace.append("Support")

			elif vspace == 1:
				y_pred_vspace.append("Refute")

			else:
				y_pred_vspace.append("Not Enough Info")


			if wmd == 0:
				y_pred_wmd.append("Support")
			
			elif wmd == 1:
				y_pred_wmd.append("Refute")

			else:
				y_pred_wmd.append("Not Enough Info")

			# print ("model label")
			# print (model.label)
			if model.label == 0:

				y_true.append("Support")

			elif model.label == 1:
				y_true.append("Refute")

			else:
				y_true.append("Not Enough Info")


			# print ("model label ", model.label)

		print (len(y_pred_vspace))
		print (len(y_true))
		# for binary, average='binary'
		# print (precision_recall_fscore_support(y_true, y_pred, average='weighted'))

		
		print ("score of tfidf ", precision_recall_fscore_support(y_true, y_pred_tfidf, average='weighted'))
		print ("accuracy of tfidf ", accuracy_score(y_true, y_pred_tfidf))

		self.save_confusion_matrix(y_true, y_pred_tfidf, "tfidf.pdf", "TF-IDF")

		print ("score of vspace ", precision_recall_fscore_support(y_true, y_pred_vspace, average='weighted'))
		print ("accuracy of vpspace ", accuracy_score(y_true, y_pred_vspace))
		self.save_confusion_matrix(y_true, y_pred_vspace, "vspace.pdf", "Vector Space")
		print ("score of wmd ", precision_recall_fscore_support(y_true, y_pred_wmd, average='weighted')) 
		print ("accuracy of wmd ", accuracy_score(y_true, y_pred_vspace))
		self.save_confusion_matrix(y_true, y_pred_wmd, "wmd.pdf", "Word Mover's Distance")

if __name__ == '__main__':

	task = 'Detection'
	tunemodel = tuneModel(task)
	# dataset_path = [{'EXP_FOLDER':'./data/google/processed_google_datasets/institution/', 
	# 				'split-ds': 'test_', 'DATASET': 'institution_process.json'}]

	dataset_path = [{'EXP_FOLDER':'./data/fever/3-class/', 
					'split-ds': 'test_', 'DATASET': 'fever_3.json'}]

	print ("dataset loaded")
	# data = tunemodel.featureExtractor.load_datafiles(dataset_path)

	# tunemodel.test_on_validation_set(data, task)

	# list_of_defactoNlps = pickle.load(open("test_bin_google_institution.pkl", "rb"))
	list_of_defactoNlps = pickle.load(open("./fever_pickle_models/detectionsave_test_defacto_model_fever3", "rb"))
	print ("computin F1 score")
	tunemodel.compute_score(list_of_defactoNlps, task)
