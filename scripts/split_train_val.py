import pickle

with open("../generated_dataset/_train_v2.1_query_details_passages", "rb") as fp:   # Unpickling
   _train_query_details_list = pickle.load(fp)

with open("../generated_dataset/_train_okapi_bm_25_top10", "rb") as fp:   # Unpickling
   _train_bm25_results = pickle.load(fp)


train_query_details_list = _train_query_details_list[0: 13000]
val_query_details_list = _train_query_details_list[13000: 15000]

train_bm25_results = _train_bm25_results[0: 13000]
val_bm25_results = _train_bm25_results[13000: 15000]


with open("../generated_dataset/train_v2.1_query_details_passages", "wb") as fp:
   pickle.dump(train_query_details_list, fp)

with open("../generated_dataset/val_v2.1_query_details_passages", "wb") as fp:
   pickle.dump(val_query_details_list, fp)

with open("../generated_dataset/train_okapi_bm_25_top10", "wb") as fp:
   pickle.dump(train_bm25_results, fp)

with open("../generated_dataset/val_okapi_bm_25_top10", "wb") as fp:
   pickle.dump(val_bm25_results, fp)
