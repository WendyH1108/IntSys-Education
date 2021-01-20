import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from data_loader import get_data_loaders
import typing


class LogisticRegressionModel(nn.Module):
    """LogisticRegressionModel is the logistic regression classifier.

    This class handles only the binary classification task.

    :param num_param: The number of parameters that need to be initialized.
    :type num_param: int
    """
    def __init__(self, num_param, loss_fn):
        ## TODO 1: Set up network
        super().__init__()
        self.function = nn.Linear(num_param, 1)
        self.loss_fn=loss_fn

    def forward(self, x):
        """forward generates the predictions for the input
        
        This function does not have to be called explicitly. We can do the
        following 
        
        .. highlight:: python
        .. code-block:: python

            model = LogisticRegressionModel(1, logistic_loss)
            predictions = model(X)
    
        :param x: Input array of shape (n_samples, n_features) which we want to
            evaluate on
        :type x: typing.Union[torch.Tensor]
        :return: The predictions on x
        :rtype: torch.Tensor
        """

        ## TODO 2: Implement the logistic regression on sample x
        return F.sigmoid(self.function(x))

class MultinomialRegressionModel(nn.Module):
    """MultinomialRegressionModel is logistic regression for multiclass prob.

    This model operates under a one-vs-rest (OvR) scheme for its predictions.

    :param num_param: The number of parameters that need to be initialized.
    :type num_param: int
    :param loss_fn: The loss function that is used to calculate "cost"
    :type loss_fn: typing.Callable[[torch.Tensor, torch.Tensor],torch.Tensor]

    .. seealso:: :class:`LogisticRegressionModel`
    """
    def __init__(self, num_param, loss_fn):
        super().__init__()
        ## TODO 3: Set up network
        # NOTE: THIS IS A BONUS AND IS NOT EXPECTED FOR YOU TO BE ABLE TO DO
        self.function = nn.Linear(num_param, 1)
        self.loss_fn=loss_fn

    def forward(self, x):
        """forward generates the predictions for the input
        
        This function does not have to be called explicitly. We can do the
        following 
        
        .. highlight:: python
        .. code-block:: python

            model = MultinomialRegressionModel(1, cross_entropy_loss)
            predictions = model(X)
    
        :param x: Input array of shape (n_samples, n_features) which we want to
            evaluate on
        :type x: typing.Union[np.ndarray, torch.Tensor]
        :return: The predictions on x
        :rtype: torch.Tensor
        """
        ## TODO 4: Implement the logistic regression on sample x
        # NOTE: THIS IS A BONUS AND IS NOT EXPECTED FOR YOU TO BE ABLE TO DO
        return F.softmax(self.function(x))

def logistic_loss(output, target):
    """Creates a criterion that measures the Binary Cross Entropy
    between the target and the output:

    The loss can be described as:

    .. math::
        \\ell(x, y) = L = \\operatorname{mean}(\\{l_1,\dots,l_N\\}^\\top), \\quad
        l_n = -y_n \\cdot \\log x_n - (1 - y_n) \\cdot \\log (1 - x_n),

    where :math:`N` is the batch size.

    Note that the targets :math:`target` should be numbers between 0 and 1.

    :param output: The output of the model or our predictions
    :type output: torch.Tensor
    :param target: The expected output or our labels
    :type target: typing.Union[torch.Tensor]
    :return: torch.Tensor
    :rtype: torch.Tensor
    """
    # TODO 2: Implement the logistic loss function from the slides using
    # pytorch operations
    length=len(output)
    return torch.sum(-torch.log(output)*target-torch.log(1-output)*(1-target))

def cross_entropy_loss(output, target):
    """Creates a criterion that measures the Cross Entropy
    between the target and the output:
    
    It is useful when training a classification problem with `C` classes.

    :param output: The output of the model or our predictions
    :type output: torch.Tensor
    :param target: The expected output or our labels
    :type target: typing.Union[torch.Tensor]
    :return: torch.Tensor
    :rtype: torch.Tensor
    """
    # NOTE: THIS IS A BONUS AND IS NOT EXPECTED FOR YOU TO BE ABLE TO DO
    length=len(output)
    return -torch.sum(torch.log(output)*target)


if __name__ == "__main__":
    # TODO: Run a sample here
    # Look at linear_regression.py for a hint on how you should do this!!

    # DS3 and DS4 should use logistic regression
    train_loader, val_loader, test_loader =get_data_loaders('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a2/data/DS3.csv', 
                                        transform_fn=None,  # Can also pass in None here
                                        train_val_test=[0.8,0.2,0.2], 
                                        batch_size=2)
    model = LogisticRegressionModel(2,logistic_loss)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for t in range(200):
        for batch_index, (input_t, y) in enumerate(train_loader):
            optimizer.zero_grad()
            preds = model(input_t)
            loss = model.loss_fn(preds, y) 
            loss.backward() 
            optimizer.step()
    model.eval()
    total_loss=0
    for batch_index, (input_t, y) in enumerate(test_loader):
      preds = model(input_t)
      total_loss=total_loss+logistic_loss(preds,y)

    print(total_loss/len(test_loader))
