from lstm.preprocess import preProcessing 
import jsonlines
import pickle

from keras.models import load_model

from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
import numpy as np

from keras.utils import np_utils

class testModel:

	def __init__(self, claims, sents, labels, model_path, dataset_name, num_classes):


		self.data_name = dataset_name
		self.model_path = model_path
		self.num_classes = num_classes
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

		with open("lstm_test_results.log", "w") as f:
			f.write("test loss "+str(loss))
			f.write("\r\n test accuracy "+ str(accuracy))
			f.write("\r\n score of lstm "+ str(precision_recall_fscore_support(self.labels, y_pred, average=avg)))


