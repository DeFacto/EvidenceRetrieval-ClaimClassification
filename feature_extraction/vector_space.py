from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from scipy import spatial

class VectorSpace:

	def __init__(self):
		pass

	def apply_vector_space(self, claim, sentences):

		vectorizer = CountVectorizer()
		vectors = vectorizer.fit_transform([claim]+[sentences]).toarray()
		return self.find_similarity(vectors[1:], vectors[0], sentences)


	def find_similarity(self, sent_vectors, claim_vector, sentences):
		
		similarities = [1 - spatial.distance.cosine(claim_vector, sent_vectors[vec]) 
								for vec in range((sent_vectors.shape)[0])]

		sorted_similarities_index = sorted(range(len(similarities)), 
										key=similarities.__getitem__, reverse=True)
		highest_similarity_score = similarities[sorted_similarities_index[0]]

		return (sentences[sorted_similarities_index[0]], round(highest_similarity_score, 2))


if __name__ == '__main__':
	
	vector_space = VectorSpace()
	claim = 'diego is a good guy'
	# relevant_sentence, claim, score = 
	relevant_sentence, score = vector_space.apply_vector_space(claim,['diego is a guy','diego is cool', 'diego is a bad guy','diego loves Portugal','diego is a researcher','some unrealted sentence'])
	print ("relevant sentence ", relevant_sentence)
	print ("claim ", claim)
	print ("score ", score)

# cosine_similarity(vector1, vector2)