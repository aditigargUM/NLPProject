import pickle

with open("../passages", "rb") as fp:   # Unpickling
    passages = pickle.load(fp)

print("Number of passages: %d" % len(passages))


with open("../train_query_passages_answer_list", "rb") as fp:   # Unpickling
    train_query_passages_answer_list = pickle.load(fp)

print("Number of queries in train dataset: %d" % len(train_query_passages_answer_list))


with open("../test_query_passages_answer_list", "rb") as fp:   # Unpickling
    test_query_passages_answer_list = pickle.load(fp)

print("Number of queries in test dataset: %d" % len(test_query_passages_answer_list))