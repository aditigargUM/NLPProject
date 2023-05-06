import pickle

# with open("../generated_dataset/passages", "rb") as fp:   # Unpickling
#     passages = pickle.load(fp)

# print("Number of passages: %d" % len(passages))


with open("../generated_dataset/train_v2.1_query_details_passages", "rb") as fp:   # Unpickling
    train_query_details = pickle.load(fp)


for i in range(5):
    print('Query:')
    print(train_query_details[i]['query'])
    print()

    print('Passages:')
    print(train_query_details[i]['passages'])
    print()

    print('Answer:')
    print(train_query_details[i]['wellFormedAnswer'])
    print()
    print()

# print("Number of queries in train dataset: %d" % len(train_query_details))


# with open("../generated_dataset/dev_v2.1_query_details_passages", "rb") as fp:   # Unpickling
#     dev_query_details = pickle.load(fp)

# print("Number of queries in dev dataset: %d" % len(dev_query_details))