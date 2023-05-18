## Contents of this README


This file contains the hyperparameter based scripts used to finetune the RAG model with DistillBERT TAS-B encoder and
T5-Small generator.


### Important note

While using the DistillBERT TAS-B encoder [link](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b), it was unable to be used in the original RAG setup in a straightforward manner. The reason behind this issue was that, when used inside a question encoder, this encoder outputs a tensor of all the word embeddings of the input query, as opposed to the expected behaviour of giving out only a single embedding representing the input query.

In order to fix this, I had to change some of the code in the core library of the rag model present in huggingface `transformers`. In modelling_rag.py at line 987:

Change the line to:
```
question_hidden_states = (self.question_encoder(input_ids, attention_mask=attention_mask)[0])[:,0,:]
```

This helps in selecting only the embeddings of the CLS token, rather than selecting all the embeddings.

Similarly, in the line 1468, change the line to:

```
question_hidden_states = (self.question_encoder(input_ids, attention_mask=attention_mask)[0])[:,0,:]
```

This helps in using the said huggingface transformer seamlessly with our RAG setup