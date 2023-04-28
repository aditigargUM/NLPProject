# NLPProject
A place to store progress of the NLP Project


# Progress:
Update [Garima] 
1. You can download the requirements.txt file and create your own conda environment with it and run the ipynb or py files to see output on a small sample of eli5 dataset.
2. You can experiment with off-the-shelf rag sequence model trained on NQ dataset
3. You can experiment with rag sequence model base with custom encoder and custom generator

So far I have generated a dataset that contains -

All QA type data from MS Marco where there is a well formed answer present for a query. More details here - https://huggingface.co/datasets/ms_marco/viewer/v2.1/validation

For the training and validation dataset stored in a pickle file, we have a structure similar to - <br/>
[{ <br/>
  &nbsp;&nbsp;&nbsp;&nbsp;'query': '', <br/>
  &nbsp;&nbsp;&nbsp;&nbsp;'passage_urls': ['', '', ''], <br/>
  &nbsp;&nbsp;&nbsp;&nbsp;'wellFormedAnswer': '' <br/>
},<br/>
...<br/>
]<br/>

For passages, we have a dictionary stored in a pickle file which contains a dictionary with _passage_url_ as the key and _passage_text_ as the value.

To load any of these files use - <br/>

with open("train_v2.1_query_details", "rb") as fp:   # Unpickling
 	query_details = pickle.load(fp)
