# Dataset:

The given dataset is obtained from - https://huggingface.co/datasets/ms_marco/viewer/v2.1/validation


For both the training and validation datasets, we have extracted queries that contain a well-formed answer and have retrieved passages for such queries from both the training dataset and the validation dataset such that none of the passages is repeated. 

The passages are saved in the passages file as a list of passages. The number of passages is - 1550080


To generate the dataset that we have been working with, please copy the training and validation files at https://huggingface.co/datasets/ms_marco/viewer/v2.1/validation into a folder named 'given_dataset' inside the 'dataset' folder. This will create training, test and validation files that contain list of entries of the format -

{
	‘query’: queryText,
	‘passages’: [passage1Text, passage2Text, …..]
	‘wellFormedAnswer’: answerText
}


# Combination of retriever-generator-decoder models:

## Retriever:
All the files for the retrievers are stored at /retriever-generator-combinations/retriever
- The script for obtaining retrieval metrics is stored at /retriever-generator-combinations/retriever/obtain_retrieval_metrics.py and accepts filepath of a list of passage indices for all test queries to evaluate the precision, recall and f1 scores.
- The code for obtaining the BM25 results is located at /retriever-generator-combinations/retriever/bm25/obtain_bm25.py. The list of bm25 indices per query can be obtained from retriever-generator-combinations/retriever/bm25/bm25_results. All the files in this folder are pickle files.
- For DPR, all code and generated files are stored at /retriever-generator-combinations/retriever/dpr.
- The code for generating query and passage embeddings and passage index is at /retriever-generator-combinations/retriever/dpr/DPR_generate_embeddings.ipynb.
- The obtained top 10 passage indices for various query-passage embedding combinations are stored at /retriever-generator-combinations/retriever/dpr/results and their performance can be directly evaluated using the ipython notebook at /retriever-generator-combinations/retriever/dpr/DPR_obtain_results.ipynb. 

## Generator:
All the files for the generators are stored at /retriever-generator-combinations/generator.
- The code for generating answers and evaluating results for various generator models is written at /retriever-generator-combinations/generator/generator.ipynb.
- However, since the answers are already generated and saved by us, the sections of the code used to generate answers are commented out.
- All the generated files for various generator-decoder combinations are stored at /retriever-generator-combinations/generator/generator_results
- The output results for each retriever-generator-decoder combination are saved in files named in the following fashion => retrieverName_generatorName_decoderName_answers. As an example, we have stored t5 answers for passages obtained from DPR retriever generated using greedy decoding at /retriever-generator-combinations/generator/generator_results/t5_generator/bm25_top10_passages/bm25_t5_greedy_answers.
- For each of these answers, we also store a corresponding analysis json file, stored at locations of the format /retriever-generator-combinations/generator/generator_results/t5_generator/bm25_top10_passages/bm25_t5_greedy_analysis.json. This file contains the list of 5 queries each of the lowest, middle and highest ROUGE-L precision, recall and f1 score. Along with the query, it also contains the corresponding well formed answers, the retrieved passages and the generated answer. These helped us perform human evaluation of various combinations of models.

# Unified RAG models:

## Indexing:
All code to create embeddings and FAISS index is in /RAG/index
- indexing-dpr is for generating embeddings and index using DPR encoders
- indexing.ipynb and indexing.py is for generating embeddings and index using SentenceTransformer for DistillBERT TAS-B model
- load_index_script.ipynb is sample script to load generated dataset and index

## RAG-POC
All code which was used to do proof-of-concept for unified RAG models intially.

/RAG/generate_rag_format_files.ipynb is to convert the dataset into input files suitable for RAG finetuning

## RAG finetuning and eval
We primarily used https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag for finetuning and evaluating our RAG models.
Some bug-fixes we did:
- Manually saved checkpoint after finetuning in /transformers/examples/research_projects/rag/lightning_base.py with "trainer.save_checkpoint("example.ckpt")" after "trainer.fit(model)" call.
- While using the DistillBERT TAS-B encoder [link](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b), it was unable to be used in the original RAG setup in a straightforward manner. The reason behind this issue was that, when used inside a question encoder, this encoder outputs a tensor of all the word embeddings of the input query, as opposed to the expected behaviour of giving out only a single embedding representing the input query.

In order to fix this, as a temporary fix, some code changes were made in the core library of the rag model present in huggingface `transformers`. In `modeling_rag.py` at line 987:

Change the line to:
```
question_hidden_states = (self.question_encoder(input_ids, attention_mask=attention_mask)[0])[:,0,:]
```

This helps in selecting only the embeddings of the CLS token, rather than selecting all the embeddings.

Similarly, in the line 1468 of `modeling_rag.py`, change the line to:

```
question_hidden_states = (self.question_encoder(input_ids, attention_mask=attention_mask)[0])[:,0,:]
```

This helps in using the said huggingface transformer seamlessly with our RAG setup

# ChatGPT as generator model:
All code to integrate with ChatGPT APIs and use it to generate outputs for the three retriever configurations are in /chatgpt
/chatgpt/chatgpt_responses has the outputs generated for these three configurations

