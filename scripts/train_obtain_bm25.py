from fastbm25 import fastbm25
import pickle
import string

def remove_punctuation(input_string):
    return input_string.translate(str.maketrans('', '', string.punctuation)).lower()

with open("../generated_dataset/train_v2.1_query_details_passages", "rb") as fp:   # Unpickling
    query_details_passages = pickle.load(fp)

with open("../generated_dataset/passages", "rb") as fp:   # Unpickling
    passages = pickle.load(fp)

tokenized_passages = [remove_punctuation(doc).split(" ") for doc in passages]

bm25 = fastbm25(tokenized_passages)

queries = [query_item['query'] for query_item in query_details_passages]
tokenized_queries = [remove_punctuation(query).split(" ") for query in queries]

top_five_passages_index_list = []
num_of_queries = len(queries)

for index, tokenized_query in enumerate(tokenized_queries):
    similarity_tuples_list = bm25.top_k_sentence(tokenized_query, k=5)
    similar_passages_index = [similarity_tuple[1] for similarity_tuple in similarity_tuples_list]
    top_five_passages_index_list.append(similar_passages_index)
    print("Done..... %d/%d" %(index, num_of_queries))



with open("../generated_dataset/train_okapi_bm_25_top5", "wb") as fp:   # Unpickling
    pickle.dump(top_five_passages_index_list, fp)
