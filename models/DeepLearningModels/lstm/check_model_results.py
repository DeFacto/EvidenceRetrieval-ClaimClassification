from lstm.preprocess import preProcessing 
import jsonlines
import pickle

from keras.models import load_model

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import np_utils

intToLabel = {}
intToLabel[0] = "Support"
intToLabel[1] = "Refute"
intToLabel[2] = "Not Enough Info"

# This script is used to test the model as well as plot the confusion matrix

class testModel:

	def __init__(self, claims, sents, labels, model_path, nb_classes):

		# self.data_name = dataset_name
		self.model_path = model_path
		self.num_classes = nb_classes
		self.claims = claims
		self.sents = sents
		self.labels = labels


	def get_results_on_test_data(self, preprocess):

		model = load_model(self.model_path)
		batch = 64		
		
		if self.num_classes == 3:

			self.labels = np_utils.to_categorical(self.labels, self.num_classes)
			avg = 'weighted'
		else:
			avg = 'binary'
		# model = load_model('model_lstm_google_birth_place_.h5')
		
		# print (model.summary())
		print ("evaluating the model ")
		print ("inside evaluation ", len(self.claims))
		print ("labels length ", len(self.labels))
		print ("sentence length  ", len(self.sents))

		loss, accuracy = model.evaluate({'claims':self.claims, 'sentences': self.sents},
		 									self.labels)

		print ("loss and accuracy evaluated")

		print ("test loss ", loss)

		print ("test accuracy ", accuracy)

		y_pred = (np.asarray(model.predict({'claims': self.claims, 'sentences': self.sents} , batch_size=batch))).round()

		print ("score of lstm ", precision_recall_fscore_support(self.labels, y_pred, average=avg)) 

		filename = "lstm_cm.pdf"
		method = "lstm"

		print ("y pred ", y_pred.shape)
		print ("type yped ", type(y_pred))

		plot_y_pred = np.argmax(y_pred, axis=1).tolist()
		plot_y_true = np.argmax(self.labels, axis=1).tolist()

		print ("plit y pred ", plot_y_pred)
		print ("pl0t y True ", plot_y_true)

		y_pred = [intToLabel[pred] for pred in plot_y_pred]

		t_labels = [intToLabel[t_pred] for t_pred in plot_y_true]

		print ("before CM function ")
		self.save_confusion_matrix(t_labels, y_pred, filename, method)

		# with open("lstm_test_results.log", "w") as f:
		# 	f.write("test loss "+str(loss))
		# 	f.write("\r\n test accuracy "+ str(accuracy))
		# 	f.write("\r\n score of lstm "+ str(precision_recall_fscore_support(self.labels, y_pred, average=avg)))


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


if __name__ == '__main__':
	
	test_model = testModel()
	model_path = "fever_simple_claims_model/model_lstm_fever_3_acc489.h5"
	model = load_model(model_path)
	print ("model sumarry ", model.summary())