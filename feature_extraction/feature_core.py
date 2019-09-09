from feature_extraction.tfidf import TFIDF
from feature_extraction.vector_space import VectorSpace
from feature_extraction.wmd import wordMoverDistance

class featureCore:

	def __init__(self, task):

		self.task = task
		self.tfidf = TFIDF()
		self.vs = VectorSpace()
		self.wmd = wordMoverDistance()


	def get_tf_idf_score(self, list_of_defactoNlps):

		# print ("nlp mdoes ", list_of_defactoNlps)
		if self.task == 'bin-classification-fever' or self.task == 'bin-classification-google':

			for model in list_of_defactoNlps:
				relevant_sentence, score  = self.tfidf.apply_tf_idf(model.claim, model.sentences)
				#0.2 
				# bin-classification-fever, 1 represents sup, ref and 0 represents nei
				# bin-classification-google classes are binary
				# 0 represents supports, 1 represents refutes
				if score >= 0.1: 
					model.method_name["tfidf"] = {self.task :{"pred_label":0}}

				else:
					model.method_name["tfidf"] = {self.task :{"pred_label":1}}

		#for fever-3 and google dataset
		# self.task == 'tri-classification-fever' or self.task == 'bin-classification-google':
		else:
			for model in list_of_defactoNlps:
				relevant_sentence, score  = self.tfidf.apply_tf_idf(model.claim, model.sentences)
				

				# classification: > 0.2 --> Yes, < 0.2 --> NEI
				# Detection: score > 0.6 --> Support, 0.2 < score < 0.6 --> Refutes, NEI < 0.2
				if score >= 0.2: 
					#detection
					# Supports
					if score > 0.3:
						# print ("score > 0.7")
						model.method_name["tfidf"] = {self.task :{"pred_label":0}}
					# refutes
					else: # REFUTES
						# print ("score < 0.7")
						model.method_name["tfidf"] = {self.task :{"pred_label":1}}
				#label as NEI	
				else:
					# print ("score < 0.05")
					model.method_name["tfidf"] = {self.task :{"pred_label":2}}

		return list_of_defactoNlps


	def get_vector_space_score(self, list_of_defactoNlps):


		if self.task == 'bin-classification-fever' or self.task == 'bin-classification-google':
			for model in list_of_defactoNlps:

				relevant_sentence, vector_space_score  = self.vs.apply_vector_space(model.claim, model.sentences)
				# bin-classification-fever, 1 represents sup, ref and 0 represents nei
				# bin-classification-google classes are binary
				# 0 represents supports, 1 represents refutes
				if vector_space_score >= 0.2:
					model.method_name["vspace"] = {self.task :{"pred_label":0}}

				else:
					model.method_name["vspace"] = {self.task :{"pred_label":1}}					
		#for fever-3 and google dataset
					
		else:

			# print ("inside else of vector_space")
			for model in list_of_defactoNlps:
				relevant_sentence, vector_space_score  = self.vs.apply_vector_space(model.claim, model.sentences)
				# classification: > 0.2 --> Yes, < 0.2 --> NEI
				# Detection: vector_space_score > 0.6 --> Support, 0.2 < vector_space_score < 0.6 --> Refutes, NEI < 0.2
				if vector_space_score >= 0.2: 
					#detection
					# Supports
					if vector_space_score > 0.3:
						model.method_name["vspace"] = {self.task :{"pred_label":0}}
					# refutes
					else: # REFUTES
						model.method_name["vspace"] = {self.task :{"pred_label":1}}
				#label as NEI	
				else:
					model.method_name["vspace"] = {self.task :{"pred_label":2}}

		
		return list_of_defactoNlps



	def get_wmd_score(self, list_of_defactoNlps):

		# print ("nlp mdoes ", list_of_defactoNlps)
		wmd_score = 0
		if self.task == 'bin-classification-fever' or self.task == 'bin-classification-google':
			# print ("inside wmd score ")
			for model in list_of_defactoNlps:
				relevant_sentence, wmd_score  = self.wmd.compute_wm_distance(model.claim, model.sentences)
				#in bin-classification google : 0 represents suports, 1 represents refutes
				if wmd_score < 0.8:
				
					model.method_name["wmd"] = {self.task :{"pred_label":0}}

				else:
					model.method_name["wmd"] = {self.task :{"pred_label":1}}		
		#for fever-3 and google dataset
		
		else:
			for model in list_of_defactoNlps:
				relevant_sentence, wmd_score  = self.wmd.compute_wm_distance(model.claim, model.sentences)
				# classification: > 0.2 --> Yes, < 0.2 --> NEI
				# Detection: wmd_score > 0.6 --> Support, 0.2 < wmd_score < 0.6 --> Refutes, NEI < 0.2
				if wmd_score <= 2.2: 
					#detection
					# Supports
					if wmd_score < 1:
						model.method_name["wmd"] = {self.task :{"pred_label":0}}
					# refutes
					else: # REFUTES
						model.method_name["wmd"] = {self.task :{"pred_label":1}}
				#label as NEI	
				else:
					model.method_name["wmd"] = {self.task :{"pred_label":2}}		

		return list_of_defactoNlps

