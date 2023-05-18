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


# Progress:
Update [Garima] 
1. You can download the requirements.txt file and create your own conda environment with it and run the ipynb or py files to see output on a small sample of eli5 dataset.
2. You can experiment with off-the-shelf rag sequence model trained on NQ dataset
3. You can experiment with rag sequence model base with custom encoder and custom generator


