# NLPProject
A place to store progress of the NLP Project


# Progress:
Update [Garima] 
1. You can download the requirements.txt file and create your own conda environment with it and run the ipynb or py files to see output on a small sample of eli5 dataset.
2. You can experiment with off-the-shelf rag sequence model trained on NQ dataset
3. You can experiment with rag sequence model base with custom encoder and custom generator


# Dataset:

The given dataset is obtained from - https://huggingface.co/datasets/ms_marco/viewer/v2.1/validation


For both the training and dev dataset, I have extracted queries that contain a well-formed answer and have retrieved passages for such queries from both the training dataset and the dataset such that none of the passages is repeated. 

The passages are saved in the passages file as a list of passages. The number of passages is - 1550080


For each of the query details files - dev_v2.1_query_details_passages and train_v2.1_query_details_passages the structure is a list of items. Each item looks like this -

{
	‘query’: queryText,
	‘passages’: [passage1Text, passage2Text, …..]
	‘wellFormedAnswer’: answerText
}

The number of queries in the training dataset is - 153725
The number of queries in the dev dataset is - 12467


While obtaining the relevant passages using FastBM25, we generate the files -
dev_okapi_bm_25_top5 and train_okapi_bm_25_top5. Each of these files contains a list of top 5 passage indices for each query, i.e. it looks something like -
[[0, 1, 20, 500, 16], [9, 0, 1, 5, 2], ….]



Using the above files for the both the dev and train datasets, the metrics obtained are -

For train dataset -
Avg. Precision: 0.34056074158392363
Avg. Recall: 0.1712744029330889
Avg F1 score: 0.22780709147780362

For dev dataset -
Avg. Precision: 0.347044196679222
Avg. Recall: 0.17464799507674492
Avg F1 score: 0.23220722921634443


All the generated files can be obtained from  - https://drive.google.com/drive/folders/1me8G_hDF4lK6sY80PGB7ubW1Tm-XPB-j?usp=sharing

and pasted into the root of the repo.
