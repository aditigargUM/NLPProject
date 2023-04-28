import pickle


with open("../generated_dataset/passages", "rb") as fp:   # Unpickling
    passages = pickle.load(fp)

with open("../generated_dataset/train_okapi_bm_25_top5", "rb") as fp:   # Unpickling
    predicted_top_5_passages_index_list = pickle.load(fp)

with open("../generated_dataset/train_v2.1_query_details_passages", "rb") as fp:   # Unpickling
   query_details_list = pickle.load(fp)


average_precision = 0
average_recall = 0
average_f1_score = 0

for query_ind in range(len(predicted_top_5_passages_index_list)):
    predicted_top_5_passages_index = predicted_top_5_passages_index_list[query_ind]
    true_top_passages = query_details_list[query_ind]['passages']

    num_correct_predictions = 0

    for predicted_passage_index in predicted_top_5_passages_index:
        for true_passage in true_top_passages:
            if passages[predicted_passage_index] == true_passage:
                num_correct_predictions = num_correct_predictions + 1
                break

    precision = num_correct_predictions / len(predicted_top_5_passages_index) if len(predicted_top_5_passages_index) != 0 else 0
    recall = num_correct_predictions / len(true_top_passages) 
    f1 = 0 if (precision == 0 and recall == 0) else ((2 * precision * recall) / (precision + recall))

    average_precision = average_precision + precision
    average_recall = average_recall + recall
    average_f1_score = average_f1_score + f1

num_queries = len(predicted_top_5_passages_index_list)
average_f1_score = average_f1_score/num_queries
average_precision = average_precision / num_queries
average_recall = average_recall / num_queries

print(average_precision)
print(average_recall)
print(average_f1_score)