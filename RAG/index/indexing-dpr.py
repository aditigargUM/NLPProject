from sentence_transformers import SentenceTransformer, models
import pickle
import jsonlines
import time
import datasets
from datasets import Features, Sequence, Value, load_dataset, Dataset, load_from_disk
import pandas as pd
import faiss
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
import numpy as np

base_path = './'

base_out_path = base_path + 'corpus/'

collection_in_file = base_path + 'passages.pickle'

doc_embeddings_file = base_out_path + 'doc_reps_dpr.pickle'

dataset_path = base_out_path + 'marco_subsampled_dataset_dpr'
index_path = base_out_path + 'marco_faiss_index_dpr'

inference_batch_size = 32

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def read_pickle(input_file):
    with open(input_file, 'rb') as f:
       out = pickle.load(f)
    f.close()
    return out

def write_pickle(output, output_file):
    with open(output_file, 'wb') as f:
       pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    
def generate_embeddings(doc_list_file, doc_embeddings_out_file, inference_batch_size):   
    starttime = time.time()
    
    (tokenizer, model) = initialize_model()
    
    model = model.to(device)
    
    with open(doc_list_file, 'rb') as f:
       doc_list = pickle.load(f)
    f.close()
    
    total_docs = len(doc_list)
    print("Total #docs: ", total_docs)
    
    document_vectors = np.zeros([total_docs, 768])
    
    num_batches = total_docs//inference_batch_size
    
    for i in range(num_batches):
        start = i * inference_batch_size
        end = (i + 1) * inference_batch_size
        input_ids = tokenizer(doc_list[start:end], padding=True, truncation=True, return_tensors="pt").to(device)
        document_vecs = model(**input_ids).pooler_output
        document_vecs = document_vecs.detach().cpu().numpy()
        document_vectors[start:end, :] = document_vecs
        #print("{s} {e} {shape}".format(s = start, e = end, shape = document_vecs.shape))
    if (total_docs % inference_batch_size) != 0:
        start = num_batches * inference_batch_size
        end = total_docs
        input_ids = tokenizer(doc_list[start:end], padding=True, truncation=True, return_tensors="pt").to(device)
        document_vecs = model(**input_ids).pooler_output
        document_vecs = document_vecs.detach().cpu().numpy()
        document_vectors[start:end, :] = document_vecs
        #print("{s} {e} {shape}".format(s = start, e = end, shape = document_vecs.shape))
        
    print('Encoded documents: {:}'.format(document_vectors.shape))
    
    ## To save in file
    with open(doc_embeddings_out_file, 'wb') as f:
       pickle.dump(document_vectors, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    
    del doc_list
    del document_vectors
    
    endtime = time.time()
    print("Total time taken in hours: " + str((endtime-starttime)/3600))
    
def initialize_model():
    tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    return (tokenizer, model)

generate_embeddings(collection_in_file, doc_embeddings_file, inference_batch_size)

embs = read_pickle(doc_embeddings_file)
docs = read_pickle(collection_in_file)
title = ["title" for i in range(len(docs))]

embs_list = []
for i in range(embs.shape[0]):
    embs_list.append(embs[i, :])
    
chunked_corpus = {"title": title, "text" : docs}
df = pd.DataFrame(chunked_corpus)
dataset = Dataset.from_pandas(df)
print(dataset)
dataset = dataset.add_column("embeddings", embs_list)
dataset.save_to_disk(dataset_path)

index = faiss.IndexHNSWFlat(embs.shape[1], 10, faiss.METRIC_INNER_PRODUCT) 
# not clear what 2nd argument is, says number of nearest neighbors
dataset.add_faiss_index("embeddings", custom_index=index)
# And save the index
dataset.get_index("embeddings").save(index_path)

