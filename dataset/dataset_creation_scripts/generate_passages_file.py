import pickle

with open("../train_query_passages_answer_list", "rb") as fp:   # Unpickling
 	train_query_details_passages = pickle.load(fp)


with open("../test_query_passages_answer_list", "rb") as fp:   # Unpickling
 	dev_query_details_passages = pickle.load(fp)


passages_set = set([])
total_length = 0

for query_item in train_query_details_passages:
	total_length += len(query_item['passages'])
	for passage in query_item['passages']:
		passages_set.add(passage)

for query_item in dev_query_details_passages:
	total_length += len(query_item['passages'])
	for passage in query_item['passages']:
		passages_set.add(passage)


passages = list(passages_set)


with open("../passages", "wb") as fp:   # Unpickling
 	pickle.dump(passages, fp)


