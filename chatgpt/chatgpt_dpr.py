import openai
import pickle
import time

openai.api_key = "" # Use your own API Key here

chat_output_dpr_file = './chatgpt_responses/output_dpr.pickle'

number_q_for_eval = 100

def read_pickle(input_file):
    with open(input_file, 'rb') as f:
       out = pickle.load(f)
    f.close()
    return out

def write_pickle(output, output_file):
    with open(output_file, 'wb') as f:
       pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    
def read_dpr(): # these are actually the test files, before segregation into train val test
    with open("dev_v2.1_query_details_passages", "rb") as fp:   # Unpickling
        query_details_passages = pickle.load(fp)

    with open("passages.pickle", "rb") as fp:   # Unpickling
        passages = pickle.load(fp)

    with open("query_marcob256_passage_marcob256_top10", "rb") as fp:   # Unpickling
        dpr_top10 = pickle.load(fp)
        
    return (query_details_passages, passages, dpr_top10)

def get_query_and_top_passages(queries, passages, dpr_top10, index):
    query = queries[index]
    passage_list = list(passages[i] for i in dpr_top10[index])
    return (query, passage_list)
    
(queries, passages, dpr_top10) = read_dpr()

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages = [
     {"role": "system", "content" : "You’re very good at comprehending reading comprehesions and finding answers to questions from them."}
    ]
)
print(completion.choices[0].message.content)

def chat_inference(chat_output_file, queries, passages, retrieved_top10, number_queries=3, ground_truth=True):
    chat_responses = []
    for index in range(number_queries):
        (query, ret_passages) = get_query_and_top_passages(queries, passages, retrieved_top10, index) 
        passage_list = query['passages']
        #print(query['wellFormedAnswer'])
        if (ground_truth == False):
            passage_list = ret_passages
        query_text = query['query']

        if len(passage_list) == 10:
            prompt = "Given a question and 10 passages, find a well formed answer sentence. Do not mention the source in the 'answer'. \n Q: {q} \n P1: {p1} \n P2: {p2} \n P3: {p3} \n P4: {p4} \n P5: {p5} \n P6: {p6} \n P7: {p7} \n P8: {p8} \n P9: {p9} \n P10: {p10} \n".format(q=query_text, p1=passage_list[0], p2=passage_list[1], p3=passage_list[2], p4=passage_list[3], p5=passage_list[4],
            p6=passage_list[5], p7=passage_list[6], p8=passage_list[7], p9=passage_list[8], p10=passage_list[9])
        elif len(passage_list) == 8:
            # there are few outlier queries with just 8 grouth truth passages
            prompt = "Given a question and 8 passages, find a well formed answer sentence. Do not mention the source in the 'answer'. \n Q: {q} \n P1: {p1} \n P2: {p2} \n P3: {p3} \n P4: {p4} \n P5: {p5} \n P6: {p6} \n P7: {p7} \n P8: {p8} \n".format(q=query_text, p1=passage_list[0], p2=passage_list[1], p3=passage_list[2], p4=passage_list[3], p5=passage_list[4],
            p6=passage_list[5], p7=passage_list[6], p8=passage_list[7])
        else:
            raise Exception("Passage list len not compatible!!")
        #print(prompt)

        input_dict = [{"role": "user", "content": prompt}]

        completion = openai.ChatCompletion.create(
          model="gpt-3.5-turbo-0301",
          messages=input_dict
        )

        chat_response = completion.choices[0].message.content
        #print(f'ChatGPT: {chat_response}')
        #print()
        chat_responses.append(chat_response)
        time.sleep(21) # imp to add this as ChatGPT has a rate limit of 3 request/min, so in every 20s
    #print(len(chat_responses))
    #assert len(chat_responses) == number_queries
    write_pickle(chat_responses, chat_output_file)
    
chat_inference(chat_output_dpr_file, queries, passages, dpr_top10, number_q_for_eval, False)

y = read_pickle(chat_output_dpr_file)
print(len(y))

y[0:2]