from gensim.models import word2vec
from pre_process import sen_tokenizer, read_data
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

num_features = 150
min_word_count = 2 
num_workers = 4     
context = 3        
downsampling = 1e-3

def clean_dataset(list1, list2):

    new_list1 =[]
    new_list2 =[]
    for i in range(len(list1)):
        if not (np.isinf(list1[i].any()) or np.isnan(list1[i]).any()):
            new_list1.append(list1[i])
            new_list2.append(list2[i])
    return np.array(new_list1), np.array(new_list2)

def featureVecMethod(words, model, num_features):
    featureVec = np.zeros(num_features,dtype="float64")
    nwords = 0
    index2word_set = set(model.wv.index2word)
    
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    
    # Dividing the result by number of words to get average
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float64")
    for review in reviews:
        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
        counter = counter+1
    print(reviewFeatureVecs)
    return reviewFeatureVecs
if __name__ == "__main__":
    X_train, y_train = read_data('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a4/data/train.csv')
    X_val, y_val = read_data('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a4/data/val.csv')
    sentences = []
    for sen in X_train:
        sentences.append(sen_tokenizer(sen))
    val_sentences = []
    for sen in X_val:
        val_sentences.append(sen_tokenizer(sen))

    model = word2vec.Word2Vec(sentences,
                            workers=num_workers,
                            size=num_features,
                            min_count=min_word_count,
                            window=context,
                            sample=downsampling,
                            iter=500)
    model.init_sims(replace=True)


    trainDataVecs = getAvgFeatureVecs(sentences, model, num_features)
    valDataVecs = getAvgFeatureVecs(val_sentences, model, num_features)
    trainDataVecs, y_train = clean_dataset(trainDataVecs, y_train)
    valDataVecs, y_val = clean_dataset(valDataVecs, y_val)

    classifier = LogisticRegression(multi_class='multinomial', max_iter=300)
    classifier.fit(trainDataVecs,y_train)
    result = classifier.predict(valDataVecs)

    print(classifier.score(trainDataVecs, y_train))
    print(classifier.score(valDataVecs, y_val))
