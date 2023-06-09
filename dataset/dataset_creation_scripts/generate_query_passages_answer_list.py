import json
import pickle

f = open('../given_dataset/dev_v2.1.json')

data = json.load(f)
print(data.keys())

## Delete all answers, query_type and query_id from the data
data.pop('query_type')
data.pop('query_id')


## Finding all the query indices with empty well formed answers
q_ind_empty_wfas = []
wfas_list = data['wellFormedAnswers']
for q_ind in wfas_list.keys():
	if (isinstance(wfas_list[q_ind], str)):
		q_ind_empty_wfas.append(q_ind)

## From the dictionary of passages, query and wellFormedAnswers, erase data for all query indices with empty wfas
[data['passages'].pop(key) for key in q_ind_empty_wfas]
[data['query'].pop(key) for key in q_ind_empty_wfas]
[data['wellFormedAnswers'].pop(key) for key in q_ind_empty_wfas]

## Store only first well formed answer for each query
wfas_list = data['wellFormedAnswers']
for q_ind in wfas_list.keys():
	if (len(wfas_list[q_ind]) > 1):
		wfas_list[q_ind] = [wfas_list[q_ind][0]]

# In a dictionary, store query_ind to {'query': '', 'passages': [], 'wellFormedAnswer': ''}
query_details_passages = []
num_dev_queries = 3000

for q_ind in data['query'].keys():
	if (len(query_details_passages) >= num_dev_queries):
		break

	passages = [passage['passage_text'] for passage in data['passages'][q_ind]]
	is_selected = [passage['is_selected'] for passage in data['passages'][q_ind]]
	selected_passage_index = is_selected.index(1) if 1 in is_selected else -1

	if selected_passage_index != -1:
		selected_passage = passages[selected_passage_index]
		query_details_passages.append({
			'query': data['query'][q_ind],
			'passages': passages,
			'wellFormedAnswer': data['wellFormedAnswers'][q_ind][0],
			'selectedPassage': selected_passage
		})

# Clearing memory
del data
f.close()

# Sanity check
print(len(query_details_passages))
# print(len(passage_url_to_text.keys()))

with open("../test_query_passages_answer_list", "wb") as fp:   #Pickling
	pickle.dump(query_details_passages, fp)


# Sanity check
with open("../test_query_passages_answer_list", "rb") as fp:   # Unpickling
 	query_details = pickle.load(fp)
 	print(len(query_details))