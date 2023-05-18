# NLPProject
A place to store progress of the NLP Project

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


# Combination of retriever-generator-decoder model:

## Retriever:
All the files for the retrievers for these models are /retriever-generator-combinations/retriever
- The script for obtaining retrieval metrics is stored at /retriever-generator-combinations/retriever/obtain_retrieval_metrics.py and accepts filepath of a list of passage indices for all test queries to evaluate the precision, recall and f1 scores.
- The code for obtaining the BM25 results is located at /retriever-generator-combinations/retriever/bm25/obtain_bm25.py. The list of bm25 indices per query can be obtained from retriever-generator-combinations/retriever/bm25/bm25_results. All the files in this folder are pickle files.
- For DPR, all code and generated files are stored at /retriever-generator-combinations/retriever/dpr.
- The code for generating query and passage embeddings and passage index is at /retriever-generator-combinations/retriever/dpr/DPR_generate_embeddings.ipynb.
- The obtained top 10 passage indices for various query-passage embedding combinations are stored at /retriever-generator-combinations/retriever/dpr/results and their performance can be directly evaluated using the ipython notebook at /retriever-generator-combinations/retriever/dpr/DPR_obtain_results.ipynb. 

# Progress:
Update [Garima] 
1. You can download the requirements.txt file and create your own conda environment with it and run the ipynb or py files to see output on a small sample of eli5 dataset.
2. You can experiment with off-the-shelf rag sequence model trained on NQ dataset
3. You can experiment with rag sequence model base with custom encoder and custom generator


