import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_loader import get_data_loaders
from typing import List, Union, Tuple
import torch.nn.functional as F


class SimpleNeuralNetModel(nn.Module):
    """SimpleNeuralNetModel [summary]
    
    [extended_summary]
    
    :param layer_sizes: Sizes of the input, hidden, and output layers of the NN
    :type layer_sizes: List[int]
    """
    def __init__(self, layer_sizes: List[int]):
        super(SimpleNeuralNetModel, self).__init__()
        # TODO: Set up Neural Network according the to layer sizes
        # The first number represents the input size and the output would be
        # the last number, with the numbers in between representing the
        # hidden layer sizes
        self.layers=[]
        for i in range(layer_sizes):
            layer1 = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer1)
            if(i != layer_sizes-1):
                layer2=nn.BatchNorm1d(layer_sizes[i+1])
                self.layers.append(layer2)
    
    def forward(self, x):
        """forward generates the prediction for the input x.
        
        :param x: Input array of size (Batch,Input_layer_size)
        :type x: np.ndarray
        :return: The prediction of the model
        :rtype: np.ndarray
        """
        for i in range(len(self.layers)):
            if(i%2!=0):
                x=self.layers[i](x)
            else:
                x=F.relu(self.layers[i](x))

        x = F.softmax(x)
        return x

class SimpleConvNetModel(nn.Module):
    """SimpleConvNetModel [summary]
    
    [extended_summary]
    
    :param img_shape: size of input image as (W, H)
    :type img_shape: Tuple[int, int]
    :param output_shape: output shape of the neural net
    :type output_shape: tuple
    """

    # the image_shape should be larger than 16*16, otherwise the parameters and number of layers should be changed
    def __init__(self, img_shape: Tuple[int, int], output_shape: tuple):
        super(SimpleConvNetModel, self).__init__()
        # TODO: Set up Conv Net of your choosing. You can / should hardcode
        # the sizes and layers of this Neural Net. The img_size tells you what
        # the input size should be and you have to determine the best way to
        # represent the output_shape (tuple of 2 ints, tuple of 1 int, just an
        # int , etc).
        self.w=img_shape[0]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 14, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(14),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(14 * (self.w//4-1) ** 2, 120),
            nn.Linear(120, 84),
            nn.Linear(84, output_shape)
        )
        
    def forward(self, x):
        """forward generates the prediction for the input x.
        
        :param x: Input array of size (Batch,Input_layer_size)
        :type x: np.ndarray
        :return: The prediction of the model
        :rtype: np.ndarray
        """
        x = self.cnn_layers(x)
        x = x.view(-1,14 * (self.w//4-1) ** 2)
        x = self.linear_layers(x)
        x = F.softmax(x)
        return x


if __name__ == "__main__":
    ## You can use code similar to that used in the LinearRegression file to
    # load and train the model.
    train_loader, val_loader, test_loader =get_data_loaders('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a3/data/data/cleaned_data.pkl', 
                                            '/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a3/data/data/cleaned_label.pkl',
                                            train_val_test=[0.8,0.2,0.2], 
                                            batch_size=20)
    model = SimpleConvNetModel([28,28],10)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.25)
    for t in range(20):
        total_loss=0
        for batch_index, (input_t, y) in enumerate(train_loader):
            optimizer.zero_grad()
            preds = model(input_t)
            loss = F.cross_entropy(preds, y)
            total_loss = total_loss + loss
            loss.backward() 
            optimizer.step()
        print(total_loss/len(test_loader))
    model.eval()
    for batch_index, (input_t, y) in enumerate(test_loader):
      preds = model(input_t)
      loss = F.cross_entropy(preds,y)
