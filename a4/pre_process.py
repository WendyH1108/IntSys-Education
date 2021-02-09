import csv
import os
import en_core_web_sm
import spacy
import string
from spacy.lang.en import English
import numpy as np
from sklearn.base import TransformerMixin
import pandas as pd

stop_words = spacy.lang.en.stop_words.STOP_WORDS
punctuations = string.punctuation
parser = English()
def sen_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    string = ''.join([i for i in sentence if i not in "1234567890"])
    mytokens = parser(string)
    # Lemmatizing each token, converting each token into lowercase, and removing space before and after word
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    newtokens = []
    for word in mytokens:
        if (word not in stop_words) and (word not in punctuations):
            newtokens.append(word)

    return newtokens

#customize transformer
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

def clean_text(text):
    return text.strip().lower()

def read_data(path):
    data=pd.read_csv(path)
    label=[]
    for item in data['label']:
        if 0<=item<=0.2: label.append(0)
        if 0.2<item<=0.4: label.append(1)
        if 0.4<item<=0.6: label.append(2)
        if 0.6<item<=0.8: label.append(3)
        if 0.8<item<=1: label.append(4)
    return data['phrase'], label