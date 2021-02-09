import numpy as np
from typing import List, Union, Tuple
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from pre_process import sen_tokenizer, predictors, read_data
import torch.optim as optim
import torch
from data_loader import get_data_loaders


class SimpleNeuralNetModel(nn.Module):
    """SimpleNeuralNetModel [summary]
    
    [extended_summary]
    
    :param layer_sizes: Sizes of the input, hidden, and output layers of the NN
    :type layer_sizes: List[int]
    """
    def __init__(self):
        super(SimpleNeuralNetModel, self).__init__()
        # TODO: Set up Neural Network according the to layer sizes
        # The first number represents the input size and the output would be
        # the last number, with the numbers in between representing the
        # hidden layer sizes
        self.layers = nn.Sequential(
                        nn.Linear(99854, 100),
                        nn.ReLU(),
                        nn.Linear(100, 50),
                        nn.Linear(50,5),
                        )
    
    def forward(self, x):
        """forward generates the prediction for the input x.
        
        :param x: Input array of size (Batch,Input_layer_size)
        :type x: np.ndarray
        :return: The prediction of the model
        :rtype: np.ndarray
        """
        x=self.layers(x)
        x = F.softmax(x)
        return x


if __name__ == "__main__":
    ## You can use code similar to that used in the LinearRegression file to
    # load and train the model.
    # X_train, y_train = read_data('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a4/data/train.csv')
    # X_test, y_test = read_data('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a4/data/test.csv')
    train_loader = get_data_loaders('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a4/data/train.csv', 
                                        batch_size=32)
    test_loader = get_data_loaders('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a4/data/test.csv', 
                                        batch_size=32)
    # tfidf_vector = TfidfVectorizer(tokenizer = sen_tokenizer)
    # tfidf_trans = TfidfTransformer()
    # #to better re-organize and understan sinppets in sentence 
    # bow_vector = CountVectorizer(tokenizer = sen_tokenizer, ngram_range=(1,2))
    # pipe = Pipeline([("preprocesser", predictors()),
    #              ("count", bow_vector),
    #              ('vectorizer', tfidf_trans)])
    # X = pipe.fit_transform(X_train).toarray()
    # X2= pipe.fit_transform(X_test).toarray()
    # feature = len(X[0])
    model = SimpleNeuralNetModel()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # for t in range(5):
    #     total_loss = 0
    #     for i in range(len(X)):
    #         optimizer.zero_grad()
    #         preds = model(torch.from_numpy(X[i]).float())
    #         loss = F.cross_entropy(preds.view(1,5), torch.tensor([y_train[i]]))
    #         total_loss = total_loss+loss
    #         loss.backward() 
    #         optimizer.step()
    #     print(total_loss)
    # correct = 0
    # model.eval()
    # for i in range(len(X2)):
    #     preds = model(torch.from_numpy(X2[i]).float())
    #     if torch.argmax(preds, dim=0) == y_test[i]:
    #         correct = correct + 1
    #         print(correct)
    # print( 100 * correct / len(X2))
    for t in range(5):
        total_loss=0
        for batch_index, (input_t, y) in enumerate(train_loader):
            optimizer.zero_grad()
            preds = model(input_t)
            loss = F.cross_entropy(preds, y) 
            total_loss = total_loss + loss
            loss.backward() 
            optimizer.step()
        print(total_loss)
    model.eval()
    total_loss=0
    for batch_index, (input_t, y) in enumerate(test_loader):
      preds = model(input_t)
      total_loss=total_loss+F.cross_entropy(preds,y)
    
