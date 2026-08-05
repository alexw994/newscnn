

# newscnn
## The dataset is from NLPCC2017 News Headline Classification
## Running Steps:
First 

> python cut_txt.py

This step performs word segmentation and saves the segmentation results.

Then

> python train_w2v.py

This step trains a word2vec model using only the corpus provided in this project. Using additional corpora would yield better results.
     
> python train.py

Finally

train a cnn classification model.
## Accuracy can reach 74%
