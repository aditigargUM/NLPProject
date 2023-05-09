from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import load_dataset
import torch

def load_data(dataset_name):
    data_files = {"train": "train.json", "validation": "validation.json", "test": "test.json"}
    data = load_dataset(dataset_name, data_files=data_files)
    return data

def get_data_subset(data, size=10):
    examples = []
    for i in range(0, size):
        examples.append(data[i]['title'] + "\n" + data[i]['selftext'])
    return examples

def initialize(model_name):
    tokenizer = RagTokenizer.from_pretrained(model_name) 
    retriever = RagRetriever.from_pretrained(model_name, index_name="compressed", use_dummy_dataset=False) # use dummy for now
    model = RagSequenceForGeneration.from_pretrained(model_name, retriever=retriever) 
    model = model.to(device)
    return (tokenizer, retriever, model)

def infer(tokenizer, retriever, model, inputs):
    input_dict = tokenizer.prepare_seq2seq_batch(inputs, return_tensors="pt").to(model.device)
    generated = model.generate(input_ids=input_dict["input_ids"]) 
    output = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return output

eli5 = load_data("vblagoje/lfqa")
eli5_train = eli5['train']
subset = get_data_subset(eli5_train, size=10)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
(tokenizer, retriever, model) = initialize("facebook/rag-sequence-nq")

output = infer(tokenizer, retriever, model, subset)
print(output)