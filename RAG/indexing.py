from sentence_transformers import SentenceTransformer, models
import pickle
import jsonlines
import time

base_path = './'

base_out_path = base_path + 'corpus/'

collection_in_file = base_path + 'passages.pickle'

doc_id_to_index_out_file = base_out_path + 'doc_id_to_index.pickle'
doc_index_to_id_out_file = base_out_path + 'doc_index_to_id.pickle'
doc_embeddings_file = base_out_path + 'doc_reps.pickle'

inference_batch_size = 512

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
    
    sentbert_model = initialize_model()
    
    with open(doc_list_file, 'rb') as f:
       doc_list = pickle.load(f)
    f.close()
    
    total_docs = len(doc_list)
    print("Total #docs: ", total_docs)
    
    # Go over documents and form sb reps for documents.
    document_vectors = sentbert_model.encode(doc_list, show_progress_bar=True, batch_size=inference_batch_size)
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
    sentbert_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
    return sentbert_model

generate_embeddings(collection_in_file, doc_embeddings_file, inference_batch_size)

