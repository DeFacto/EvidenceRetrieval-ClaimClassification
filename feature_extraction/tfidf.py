from sklearn.feature_extraction.text import TfidfVectorizer
# from preprocessing.pre_process import TextNormalization
from scipy import spatial


class TFIDF:

	def __init__(self):
		pass

	def apply_tf_idf(self, claim, sentences):
		
			vectorizer = TfidfVectorizer()
			tfidf_matrix = vectorizer.fit_transform([claim]+[sentences]).toarray()
			return self.find_similarity(tfidf_matrix[1:], tfidf_matrix[0], sentences)
			

	def find_similarity(self, sent_vectors, claim_vector, sentences):


		similarities = [1 - spatial.distance.cosine(claim_vector, sent_vectors[vec]) 
								for vec in range((sent_vectors.shape)[0])]
		
		# print ("similarities ", similarities)
		sorted_similarities_index = sorted(range(len(similarities)), 
								key=similarities.__getitem__, reverse=True)

		# print ("sorted similarities index", sorted_similarities_index)
		highest_similarity_score = similarities[sorted_similarities_index[0]]

		# print ("highest similarity ", highest_similarity_score)
		return (sentences[sorted_similarities_index[0]], round(highest_similarity_score, 2))


if __name__ == "__main__":

	tf_idf = TFIDF()
	claim = 'diego is a good guy'
	relevant_sentence, score = tf_idf.apply_tf_idf(claim,['diego is a guy','diego is cool', 'diego is a bad guy','diego loves Portugal','diego is a researcher','some unrealted sentence'])
	