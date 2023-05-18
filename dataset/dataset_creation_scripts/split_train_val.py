import pickle

with open("../_train_query_passages_answer_list", "rb") as fp:   # Unpickling
   _train_query_passages_answer_list = pickle.load(fp)

train_query_passages_answer_list = _train_query_passages_answer_list[0: 13000]
val_query_passages_answer_list = _train_query_passages_answer_list[13000: 15000]

with open("../train_query_passages_answer_list", "wb") as fp:
   pickle.dump(train_query_details_list, fp)

with open("../val_query_passages_answer_list", "wb") as fp:
   pickle.dump(val_query_details_list, fp)