import pandas as pd
import numpy as np
import pickle
import jsonlines
# from bert_serving.client import BertClient
import gzip

# from keras.layers import Concatenate, Dense, LSTM, Input, concatenate


# dataset_name = "fever_full_binary_dev"

def get_encoding():

    #dataset_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/fever_full_binary_dev_sent_ret.jsonl"
    # dataset_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/complete_pipeline/sent_ret/fever_full_binary_dev_bert.jsonl"

    dataset_path = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/fever_blind_set/fever_blind_sent_ret_dl_models.jsonl"
    claims = []
    sents = []
    labels = []

    with jsonlines.open(dataset_path, mode='r') as f:
    	tmp_dict = {}
    	for example in f:
    	    claims.append(example["claim"])
    	    sents.append(example["sentence"])
    	    # labels.append(example["label"])

    	tmp_dict = {'claim':claims, 'sentence':sents}
    	train_data = pd.DataFrame(data=tmp_dict)

    print (train_data.shape)

    claims = train_data["claim"].tolist()
    sents = train_data["sentence"].tolist()

    print ("claims length ", len(claims))

    sents_pair = [[claim+' ||| '+sent] for claim,sent in zip(claims,sents)]

    print ("sent pair length ", len(sents_pair))

    vec = np.empty((len(sents_pair), 768))

    count = 0
    batch_size = 4
    for i in range(len(sents_pair)):
        
        if not (i % batch_size):

            print (" i:i+batch_size] ", str(i)+ str(i+batch_size))
            print (sents_pair[i:i+batch_size])

            print ("\n ")
        count += 1

        if count  > 5:
            break


if __name__ == '__main__':
    
    get_encoding()
    #     if count == 0:
    #         # pass
    #         vec = bc.encode(sent)
    #     else:
    #         # pass
    #         vec = np.vstack((vec, bc.encode(sent)))
            
    #     if count % 300 == 0:
    #         print ("count ", count)
    #     count += 1

    # print ("saving vector into zip")

    # file_name = "/scratch/kkuma12s/new_embeddings/fever_full_dev_claim_cls_bert"

    # save_dataset_and_compress(vec, file_name)