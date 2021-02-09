import csv
import os
import en_core_web_sm
import spacy
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pre_process import sen_tokenizer, predictors, read_data
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline



# Here in this file, you should define functions to try out different encodings.
# Some options:
#   1) Bag of words. You don't need any library for this, it's simple enough to
#       implement on your own.
#   2) Word embeddings. You can use spacy or Word2Vec (or others, but these are good
#       starting points). Spacy will give you better embeddings, since they are 
#       already defined. Word2Vec will create embeddings based on your dataset.

## Document the choices you make, including if you pre-process words/tokens by 
# stemming, using POS (parts of speech info) or anything else. 

## Create your own files to define Logistic regression/ Neural networks to try
# out the performace of different ML algorithms. It'll be up to you to evaluate
# the performance.


class SentimentDataset(Dataset):
    """SentimentDataset [summary]
    
    [extended_summary]
    
    :param path_to_data: Path to dataset directory
    :type path_to_data: str
    """
    def __init__(self, path_to_data):
        ## TODO: Initialise the dataset given the path to the dataset directory.
        ## You may want to include other parameters, totally your choice.
        self.path = path_to_data
        self.x ,self.y = read_data(self.path)
        tfidf_vector = TfidfVectorizer(tokenizer = sen_tokenizer)
        tfidf_trans = TfidfTransformer()
        #to better re-organize and understan sinppets in sentence 
        bow_vector = CountVectorizer(tokenizer = sen_tokenizer, ngram_range=(1,2))
        pipe = Pipeline([("preprocesser", predictors()),
                    ("count", bow_vector),
                    ('vectorizer', tfidf_trans)])
        self.x = pipe.fit_transform(self.x).toarray()
    def __len__(self):
        """__len__ [summary]
        
        [extended_summary]
        """
        ## TODO: Returns the length of the dataset.
        return len(self.x)

    def __getitem__(self, index):
        """__getitem__ [summary]
        
        [extended_summary]
        
        :param index: [description]
        :type index: [type]
        """
        ## TODO: This returns only ONE sample from the dataset, for a given index.
        ## The returned sample should be a tuple (x, y) where x is your input 
        ## vector and y is your label
        ## Before returning your sample, you should check if there is a transform
        ## sepcified, and pply that transform to your sample
        # Eg:
        # if self.transform:
        #   sample = self.transform(sample)
        ## Remember to convert the x and y into torch tensors.
        self.new_x = torch.from_numpy(self.x[index]).float()
        return (self.new_x, torch.tensor(self.y[index]))

def get_data_loaders(path_to_train, batch_size=32):
    """
    You know the drill by now.
    """
    dataset = SentimentDataset(path_to_train)

    # Now, we define samplers for each of the train, val and test data
    train_loader = DataLoader(dataset, batch_size=batch_size)

    return train_loader