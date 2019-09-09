import os
import pandas as pd
import keras
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import argparse
import pickle
import sys
import jsonlines
import gzip
from random import randint

from allennlp.commands.elmo import ElmoEmbedder

class preProcessing():

    '''
    token_type can be claim or sentence
    this function takens all the claims/sentences and perform tokenization
    '''

    def tokenize_data(self, data, token_type, max_tokens=None):

        tokens_list = [text_to_word_sequence(data[str(token_type)].iloc[i], lower=False) for i in range(len(data))]
       
        return tokens_list


    def create_elmo_embeddings(self, elmo, claims_tokens, sents_tokens, documents, dataset_name=None):

        claim_embeddings = []
        sentence_embeddings = []
        labels = []

        documentIdx = 0
        for elmo_embedding in elmo.embed_sentences(claims_tokens):  
            
            claim_document = documents["claim"].iloc[documentIdx]
            # Average the 3 layers returned from ELMo
            avg_elmo_embedding = np.average(elmo_embedding, axis=0)

            claim_embeddings.append(avg_elmo_embedding)
            #  since this script is common for all sets
            # therefore if statement is added as there are no label in fever_full_binary_fev_elmo
            if dataset_name != "fever_full_binary_dev_elmo" and dataset_name != "fever_blind_set":        
                labels.append(documents['label'].iloc[documentIdx])

            documentIdx += 1
            
            if documentIdx % 3000 ==0:
                print ("documents count ", documentIdx)

        
        documentIdx = 0
        batch_size = 16
        # embed_sentences(tokens, batch_size)
        for elmo_embedding in elmo.embed_sentences(sents_tokens, batch_size):  
            
            if dataset_name == "birth_place" or dataset_name == "institution":
                sent_document = documents["body"].iloc[documentIdx]
            else:
                sent_document = documents["sentence"].iloc[documentIdx]

            try:
        
                # Average the 3 layers returned from ELMo
                avg_elmo_embedding = np.average(elmo_embedding, axis=0)
            #because some sents have just punc ' (' due to which there is no embeddings
            except ZeroDivisionError:
                random_number = randint(4,15)
                avg_elmo_embedding = np.zeros((random_number, 1024)) 
                
            sentence_embeddings.append(avg_elmo_embedding)

            documentIdx += 1

            if documentIdx % 1000 ==0:
                print ("documents count ", documentIdx)
                
            
        return claim_embeddings, sentence_embeddings, labels

    
    def load_datafiles(self, dataset_params):


        data = dict()
        for p in dataset_params:
            open_data = open(p['EXP_FOLDER'] + p['DATASET'], "rb")
            dataframe = pickle.load(open_data)
            data[str(p['DATASET'][0:-4])] = dataframe # keys are dataset names w/o extension

        return data

    def save_dataset_and_compress(self, dataset_dict, name):

        with gzip.GzipFile(name + '.pgz', 'w') as f:
            pickle.dump(dataset_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


    def to_padding(self, claims, sentences, labels, max_claims_length, max_sents_length):

        claims_data = pad_sequences(claims, maxlen=max_claims_length)  #returns array of data
        sents_data = pad_sequences(sentences, maxlen=max_sents_length)

        return (claims_data, sents_data)



if __name__ == '__main__':

#     dataset_name = 'test_fever_3'
    dataset_name = 'fever_blind_set'
    # dataset_path = [{'EXP_FOLDER': './lstm/' , 'DATASET': 'train_fever_rej.pkl'}],
#     dataset_path = [{'EXP_FOLDER': './datasets/' , 'DATASET': 'train_'+str(dataset_name)+'.pkl'}]
    
    
    preprocess = preProcessing()
    
    
    if dataset_name == 'fever_full_binary_train' or dataset_name == 'fever_full_binary_dev':
        
        dataset_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/"+dataset_name+".jsonl"
        with jsonlines.open(dataset_path, mode='r') as f:
            claims = []
            sents = []
            labels = []
            for example in f:
                claims.append(example["claim"])
                sents.append(example["sentence"])
                labels.append(example["label"])
            
            tmp_dict = {'claim':claims, 'sentence':sents, 'label': labels}
            dataframe = pd.DataFrame(data=tmp_dict)
                
    else:
        # print (len(dataset["train_fever_rej"]))
        # print (dataset["train_fever_rej"]["sentence"].iloc[0])

        if dataset_name == "fever_blind_set":
            print ("fever blind set selected")
            dataset_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/fever_blind_sent_ret_dl_models.jsonl"
            with jsonlines.open(dataset_path, mode='r') as f:

                claims = []
                sents = []
                # labels = []

                for example in f:
                    claims.append(example["claim"])
                    sents.append(example["sentence"])
                    # labels.append(example["label"])

                tmp_dict = {'claim':claims, 'sentence':sents}
                dataframe = pd.DataFrame(data=tmp_dict)

        else:

            dataset_path = [{'EXP_FOLDER': './datasets/' , 'DATASET': str(dataset_name)+'.pkl'}]
            dataset = preprocess.load_datafiles(dataset_path)
            dataframe = dataset[str(dataset_name)]

            if dataset_name == 'birth_place' or dataset_name == 'institution':

                dataframe["claim"]= dataframe["claim"].apply(lambda x: ','.join(map(str, x)))
                max_length_claims = max([len(i.split()) for i in dataframe["claim"].tolist()])
                max_length_sents = max([len(i.split()) for i in dataframe["body"].tolist()])
                print (max_length_claims)
                print (max_length_sents)

    claims_tokens = preprocess.tokenize_data(dataframe, "claim")
    sents_tokens = preprocess.tokenize_data(dataframe, "sentence")

    
    elmo = ElmoEmbedder(cuda_device=1)

    claim_embeddings, sent_embeddings, _ = preprocess.create_elmo_embeddings(elmo, claims_tokens, sents_tokens, dataframe, dataset_name)
    preprocess.save_dataset_and_compress(claim_embeddings, "/scratch/kkuma12s/elmo_embeddings/claims_embeddings_fever_blind_sent_ret")
    # preprocess.save_dataset_and_compress(sent_embeddings, "/scratch/kkuma12s/elmo_embeddings/sents_embeddings_fever_blind_sent_ret")
    # pickle.dump(claim_embeddings, open("./embeddings/test_claim_elmo_emb_fever_full_binary_dev.pkl", "wb"))
    # pickle.dump(sent_embeddings, open("./embeddings/test_sents_elmo_emb_fever_full_binary_dev.pkl", "wb"))
    # max_claims_length = 65
    # max_sents_length = 300
    # claims_data, sents_data, labels = preprocess.to_padding(claim_embeddings, sent_embeddings, train_label, max_claims_length, max_sents_length)

