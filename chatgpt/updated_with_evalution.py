# %%
!pip install datasets
!pip install evaluate
!pip install rouge-score

# %%
import nltk

def calculate_bleu(gold_file, predicted_file):
  sum = 0
  count = 0

  with open(gold_file, 'r') as f1, open(predicted_file, 'r') as f2:
    gold = f1.readlines()
    pred = f2.readlines()

    count = len(gold)
    for i in range(count):
      sum += nltk.translate.bleu_score.sentence_bleu(references=[gold[i].split()], hypothesis=pred[i].split())
      count += 1

    return sum/count
# print(calculate_bleu('/content/a.txt', '/content/b.txt'))

# %%
import evaluate

meteor = evaluate.load('meteor')

def calculate_meteor(gold_file, predicted_file):
  sum = 0
  count = 0

  with open(gold_file, 'r') as f1, open(predicted_file, 'r') as f2:
    gold = f1.readlines()
    pred = f2.readlines()

    return meteor.compute(predictions=pred, references=gold)['meteor']

# print(calculate_meteor('/content/a.txt', '/content/b.txt'))

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
from rouge_score import rouge_scorer

import json

def calculate_rougeL(gold_file, predicted_file):
  sum_p, sum_r, sum_f1 = 0, 0, 0
  count = 0

  data = []

  scores_ = []

  scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

  with open(gold_file, 'r') as f1, open(predicted_file, 'r') as f2:
    gold = f1.readlines()
    pred = f2.readlines()

    count = len(gold)
    # print(count)
    for i in range(count):
      scores = scorer.score(prediction=pred[i], target=gold[i])
      sum_f1 += scores['rougeL'].fmeasure
      sum_p += scores['rougeL'].precision
      sum_r += scores['rougeL'].recall
      count += 1

      scores_.append(scores)

      (prec_, rec_, f1_) = normalized_scores(pred[i], gold[i])

      data.append({
          'generated': pred[i],
          'target': gold[i],
          'precision': scores['rougeL'].precision,
          'recall': scores['rougeL'].recall,
          'fmeasure': scores['rougeL'].fmeasure,
          'normalized_precision': prec_,
          'normalized_recall': rec_,
          'normalized_f1': f1_

      })

      sum_p += prec_
      sum_r += rec_
      sum_f1 += f1_

    print('normalized precision:', sum_p/count)
    print('normalized recall:', sum_r/count)
    print('normalized f1:', sum_f1/count)

    return data
with open('/content/eval.json', 'w') as file:
  json.dump(calculate_rougeL('/content/test_ref.target', '/content/test_preds.txt'), file)

# %%



