import pandas as pd
import numpy as np
import pickle
import jsonlines
from bert_serving.client import BertClient
import gzip

# from keras.layers import Concatenate, Dense, LSTM, Input, concatenate


# dataset_name = "fever_full_binary_dev"

def get_encoding():

    #dataset_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/fever_full_binary_dev_sent_ret.jsonl"
    dataset_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/fever_full_binary_dev_bert.jsonl"

    claims = []
    sents = []
    labels = []

    with jsonlines.open(dataset_path, mode='r') as f:
    	tmp_dict = {}
    	for example in f:
    	    claims.append(example["claim"])
    	    sents.append(example["sentence"])
    	    labels.append(example["label"])

    	tmp_dict = {'claim':claims, 'sentence':sents, 'label':labels}
    	train_data = pd.DataFrame(data=tmp_dict)

    print (train_data.shape)

    # len(train_df["sentence"])
    bc = BertClient()

    claims = train_data["claim"].tolist()
    sents = train_data["sentence"].tolist()

    print ("claims length ", len(claims))

    sents_pair = [[claim+' ||| '+sent] for claim,sent in zip(claims,sents)]

    print ("sent pair length ", len(sents_pair))

    vec = np.empty((len(sents_pair), 768))

    count = 0
    for sent in sents_pair:
        
        if count == 0:
            # pass
            vec = bc.encode(sent)
        else:
            # pass
            vec = np.vstack((vec, bc.encode(sent)))
            
        if count % 300 == 0:
            print ("count ", count)
        count += 1

    print ("saving vector into zip")

    file_name = "/scratch/kkuma12s/new_embeddings/fever_full_dev_claim_cls_bert"

    save_dataset_and_compress(vec, file_name)
    # with gzip.GzipFile(file_name + '.pgz', 'w') as f:
    #     pickle.dump(vec, f, protocol=pickle.HIGHEST_PROTOCOL)



def save_dataset_and_compress(dataset_dict, name):
    with gzip.GzipFile(name + '.pgz', 'w') as f:
        pickle.dump(dataset_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_compressed_pickle_file(pickle_file_name):
    with gzip.open(pickle_file_name+'.pgz', 'rb') as f:
        return pickle.load(f)

# save_dataset_and_compress(vec, file_name)
# print ("saved **** ")

if __name__ == '__main__':

    # get_encoding()

    d1 = load_compressed_pickle_file("/scratch/kkuma12s/new_embeddings/fever_full_dev_claim_cls_bert")
    # d2 = load_compressed_pickle_file("/scratch/kkuma12s/new_embeddings/fever_full_dev_binary_sent_ret_bert_60k_120k")

    print ("shape of d1", d1.shape)
    # print ("shape of d2", d2.shape)
    # combine = np.concatenate((d1, d2))
    # # save_dataset_and_compress(combine, "/scratch/kkuma12s/new_embeddings/fever_full_dev_binary_sent_ret_bert_120k_complete")
    # print ("saved combined d1 and d2")

    # d3 = load_compressed_pickle_file("/scratch/kkuma12s/new_embeddings/fever_full_dev_binary_sent_ret_bert_120k_plus")
    # print ("combined shape ", combine.shape)
    # print ("shape of d3 ", d3.shape)
    # final = np.concatenate((combine, d3))
    # print ("final shape ", final.shape)
    # print ("saving daataset ")
    # save_dataset_and_compress(final, "/scratch/kkuma12s/new_embeddings/fever_full_dev_binary_sent_ret_bert_complete")
    # print ("saved dataset")

# print ("loading the vector")

# result = load_compressed_pickle_file(file_name)
# print ("result shape ", result.shape)
