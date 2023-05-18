from fastbm25 import fastbm25
import pickle
import string

def remove_punctuation(input_string):
    return input_string.translate(str.maketrans('', '', string.punctuation)).lower()

with open("../../dataset/train_query_passages_answer_list", "rb") as fp:   # Unpickling
    train_query_passages_answer_list = pickle.load(fp)

with open("../../dataset/passages", "rb") as fp:   # Unpickling
    passages = pickle.load(fp)

tokenized_passages = [remove_punctuation(doc).split(" ") for doc in passages]

bm25 = fastbm25(tokenized_passages)

queries = [query_item['query'] for query_item in train_query_passages_answer_list]
tokenized_queries = [remove_punctuation(query).split(" ") for query in queries]

top_ten_passages_index_list = []
num_of_queries = len(queries)

for index, tokenized_query in enumerate(tokenized_queries):
    similarity_tuples_list = bm25.top_k_sentence(tokenized_query, k=10)
    similar_passages_index = [similarity_tuple[1] for similarity_tuple in similarity_tuples_list]
    top_ten_passages_index_list.append(similar_passages_index)
    print("Done..... %d/%d" %(index, num_of_queries))



with open("./bm25_results/train_okapi_bm_25_top10", "wb") as fp:   # Unpickling
    pickle.dump(top_ten_passages_index_list, fp)
