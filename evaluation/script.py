# %%
import openai
import pickle
import time

# %%
openai.api_key = "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" # Use your own API Key here

# %%
chat_output_file = './chatgpt_responses/output.pickle'
chat_output_bm_25_file = './chatgpt_responses/output_bm25.pickle'

number_q_for_eval = 100

# %%
def read_pickle(input_file):
    with open(input_file, 'rb') as f:
       out = pickle.load(f)
    f.close()
    return out

def write_pickle(output, output_file):
    with open(output_file, 'wb') as f:
       pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

# %%
def read_bm25(): # these are actually the test files, before segregation into train val test
    gen_dat = 'generated_dataset/'
    with open(gen_dat+"test_v2.1_query_details_passages", "rb") as fp:   # Unpickling
        query_details_passages = pickle.load(fp)

    with open(gen_dat+"passages", "rb") as fp:   # Unpickling
        passages = pickle.load(fp)

    with open(gen_dat+"test_okapi_bm_25_top10", "rb") as fp:   # Unpickling
        dev_okapi_bm_25_top10 = pickle.load(fp)

    return (query_details_passages, passages, dev_okapi_bm_25_top10)

def get_query_and_top_passages(queries, passages, bm25_top10, index):
    query = queries[index]
    passage_list = list(passages[i] for i in bm25_top10[index])
    return (query, passage_list)

(queries, passages, bm25_top10) = read_bm25()

# %%
print(queries[:100], bm25_top10[:100])

# %%
"""
for psg in queries[0]['passages']:
    print(psg)
    print()
"""

# %%
"""
(query, passage_list) = get_query_and_top_passages(queries, passages, bm25_top10, 2)
print(query['query'])
print()
print(query['wellFormedAnswer'])
print()
print(query['selectedPassage'])
print()
for p in passage_list[0:10]:
    print(p)
    print()
"""

# %%
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages = [
     {"role": "system", "content" : "You’re very good at comprehending reading comprehesions and finding answers to questions from them."}
    ]
)
print(completion.choices[0].message.content)

# %%
def chat_inference(chat_output_file, queries, passages, bm25_top10, number_queries=3, ground_truth=True):
    chat_responses = []
    for index in range(number_queries):
        (query, ret_passages) = get_query_and_top_passages(queries, passages, bm25_top10, index) 
        passage_list = query['passages']
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
    print(len(chat_responses))
    assert len(chat_responses) == number_queries
    write_pickle(chat_responses, chat_output_file)

# %%
chat_inference(chat_output_bm_25_file, queries, passages, bm25_top10, number_q_for_eval, False)

# %%
y = read_pickle(chat_output_bm_25_file)

# %%
len(y)

# %%
y[0:10]

# %%
def read_dpr(): # these are actually the test files, before segregation into train val test
    gen_dat = 'generated_dataset/'
    with open(gen_dat + "test_v2.1_query_details_passages", "rb") as fp:   # Unpickling
        query_details_passages = pickle.load(fp)

    with open(gen_dat+"passages", "rb") as fp:   # Unpickling
        passages = pickle.load(fp)

    with open(gen_dat+"query_marcob256_passage_marcob256_top10", "rb") as fp:   # Unpickling
        dpr_top10 = pickle.load(fp)
        
    return (query_details_passages, passages, dpr_top10)

def get_query_and_top_passages_dpr(queries, passages, dpr_top10, index):
    query = queries[index]
    passage_list = list(passages[i] for i in dpr_top10[index])
    return (query, passage_list)
    
(queries_dpr, passages_dpr, dpr_top10) = read_dpr()

# %%
from collections import Counter
import re
import string

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalized_scores(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

# %%
import json
data = []
with open('../chatgpt_responses/output_bm25.pickle', 'rb') as file, open('../chatgpt_responses/output.pickle', 'rb') as file2, open('../chatgpt_responses/output_dpr.pickle', 'rb') as dpr:
    out = pickle.load(file)
    out2 = pickle.load(file2)
    out3 = pickle.load(dpr)
    file.close()
    file2.close()
    dpr.close()
    print(len(out))
    for i in range(100):
        query, psg_lst = get_query_and_top_passages(queries=queries, passages=passages, bm25_top10=bm25_top10, index=i)
        query_dpr, psg_lst_dpr = get_query_and_top_passages_dpr(queries=queries_dpr, passages=passages_dpr, dpr_top10=dpr_top10, index=i)
        data1 = {
            'query': query,
            'psgs': psg_lst,
            'gpt_bm25': out[i],
            'gpt_gold': out2[i],
            'psgs_dpr': psg_lst_dpr,
            'gpt_dpr': out3[i]
        }
        data.append(data1)

json.dumps(data)

# %%
import pickle

# %%
with open('../chatgpt_responses/output.pickle', 'rb') as file:
    out = pickle.load(file)
    file.close()
    print(out)

# %%
# !pip install -r rouge/requirements.txt
# !pip install rouge-score
from rouge_score import rouge_scorer
def compute_rouge_score(wellFormedAnswers, test_output):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_precision_scores = []
    rouge_recall_scores = []
    rouge_f1_scores = []
    scores = []

    n_p = 0
    n_r = 0
    n_f = 0
    for ind in range(len(test_output)):
        score = scorer.score(wellFormedAnswers[ind], test_output[ind])
        pp, rr, ff = normalized_scores(wellFormedAnswers[ind], test_output[ind])
        n_p += pp
        n_r += rr
        n_f += ff
        scores.append(score)
        rouge_precision_scores.append(score['rougeL'].precision)
        rouge_recall_scores.append(score['rougeL'].recall)
        rouge_f1_scores.append(score['rougeL'].fmeasure)
    
    print(n_p/len(test_output), n_r/len(test_output), n_f/len(test_output))
    return(rouge_precision_scores, rouge_recall_scores, rouge_f1_scores)

# %%
wfa, bm25, gld, dpr_ = [], [], [], []
for i in data:
    wfa.append(i['query']['wellFormedAnswer'])
    bm25.append(i['gpt_bm25'])
    gld.append(i['gpt_gold'])
    dpr_.append(i['gpt_dpr'])

# %%
prec_bm, recall_bm, f1_bm, bm25_scores = compute_rouge_score(wfa, bm25)
prec_gl, recall_gl, f1_gl, gld_scores = compute_rouge_score(wfa, gld)
prec_dpr, recall_dpr, f1_dpr, dpr_scores = compute_rouge_score(wfa, dpr_)

# %%
print(bm25_scores)

# %%
print(gld_scores)

# %%
import numpy as np

prec_bm = np.array(prec_bm)
prec_gl = np.array(prec_gl)
prec_dpr = np.array(prec_dpr)
recall_bm = np.array(recall_bm)
recall_gl = np.array(recall_gl)
recall_dpr = np.array(recall_dpr)
f1_bm = np.array(f1_bm)
f1_gl = np.array(f1_gl)
f1_dpr = np.array(f1_dpr)

# %%
print('prec bm25:', np.mean(prec_bm), '\nprec gold: ', np.mean(prec_gl), '\nrecall bm25: ', np.mean(recall_bm), '\nrecall gold: ', np.mean(recall_gl), '\nf1 bm25: ', np.mean(f1_bm), '\nf1 gold: ', np.mean(f1_gl), '\nprec dpr: ', np.mean(prec_dpr), '\nrecall dpr: ', np.mean(recall_dpr), '\nf1 dpr: ', np.mean(f1_dpr))

# %%