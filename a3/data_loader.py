import csv
import torch
import os
import pandas as pd
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from clean_data import load_pickle_file
import torchvision.transforms as transforms
import pickle


# from clean_data import main


class SimpleDataset(Dataset):
    """SimpleDataset [summary]
    
    [extended_summary]
    
    :param path_to_pkl: Path to PKL file with Images
    :type path_to_pkl: str
    :param path_to_labels: path to file with labels
    :type path_to_labels: str
    """
    def __init__(self, path_to_pkl, path_to_labels, df=None):
        ## TODO: Add code to read csv and load data. 
        ## You should store the data in a field.
        # Eg (on how to read .csv files):
        # with open('path/to/.csv', 'r') as f:
        #   lines = ...
        ## Look up how to read .csv files using Python. This is common for datasets in projects.
        #with open(path_to_pkl, 'r') as f:
        self.df = load_pickle_file(path_to_pkl)
        np.random.shuffle(self.df)
        self.label = load_pickle_file(path_to_labels)
        np.random.shuffle(self.label)

    def __len__(self):
        """__len__ [summary]
        
        [extended_summary]
        """
        ## TODO: Returns the length of the dataset.
        return len(self.df)

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
        x=self.df[index]
        y=self.label[index]
        # transform pil.Image.Image into tensor type
        pil2tensor = transforms.ToTensor()
        x = pil2tensor(x)
        x=np.asarray(x)
        x=torch.tensor(x)
        y = torch.tensor(y)
        sample=(x,y)
        return (sample)


def get_data_loaders(path_to_pkl, 
                     path_to_labels,
                     train_val_test=[0.8, 0.2, 0.2], 
                     batch_size=32):
    """get_data_loaders [summary]
    
    [extended_summary]
    
    :param path_to_csv: [description]
    :type path_to_csv: [type]
    :param train_val_test: [description], defaults to [0.8, 0.2, 0.2]
    :type train_val_test: list, optional
    :param batch_size: [description], defaults to 32
    :type batch_size: int, optional
    :return: [description]
    :rtype: [type]
    """
    # First we create the dataset given the path to the .csv file
    dataset = SimpleDataset(path_to_pkl, path_to_labels)

    # Then, we create a list of indices for all samples in the dataset.
    dataset_size = len(dataset)
    # indices = list(range(dataset_size))

    ## TODO: Rewrite this section so that the indices for each dataset split
    ## are formed. You can take your code from last time

    ## BEGIN: YOUR CODE
    size_train = int(train_val_test[0]*dataset_size)
    size_validation = int(train_val_test[1]*size_train)
    # np.random.shuffle(dataset)
    val_indices=[0,size_validation]
    train_indices =[size_validation,size_train]
    test_indices =[size_train,-1]
    ## END: YOUR CODE
    train_set = open('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a3/data/data/train/train_data.pkl', 'wb')
    pickle.dump(dataset.df[:size_validation], train_set)
    train_set.close()
    val_set = open('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a3/data/data/val/val_data.pkl', 'wb')
    pickle.dump(dataset.df[size_validation:size_train],val_set)
    val_set.close()

    # Now, we define samplers for each of the train, val and test data
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)


    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)


    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader
