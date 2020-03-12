import torch

""" PREDICTION """
#define w =2  and b = -1, for y = wx + b
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-1., requires_grad=True)

#function forward(x) for prediction
def forward(x):
    yhat = w * x + b
    return yhat
x = torch.tensor([[1.]])
yhat = forward(x)
print("The prediction:  ", yhat)

#create x Tensor and check the shape of x tensor

x = torch.tensor([[1.0], [2.0]])
print("The shape of x: ", x.shape)

#make the prediction of y = 2*x -1 at x = [1, 2]
yhat = forward(x)
print("The prediction: ", yhat)

""" PRACTICE """
x = torch.tensor([[1.0], [2.0], [3.0]])
yhat = forward(x)
print("My result: ", yhat)


""" CLASS LINEAR """
# import class linear
from torch.nn import Linear
torch.manual_seed(1)

# create Linear Regression Model, and print out the parameters

lr = Linear(in_features=1, out_features=1, bias=True)
print("Parameters w and b: ", list(lr.parameters()))

#make the prediction at x = [[1.]]
x = torch.tensor([[1.0]])
yhat = lr(x)
print("The prediction: ", yhat)

# create the prediction using Linear Model
x = torch.tensor([[1.0], [2.0]])
yhat = lr(x)
print("The prediction: ", yhat)

""" PRACTICE """
x = torch.tensor([[1.0], [2.0], [3.0]])
yhat = lr(x)
print("THe prediction: ", yhat)


""" BUILD CUSTOM MODULES """

# library for this section
from torch import nn
#customize Linear regression class
class LR(nn.Module):
    #Constructor
    def __init__(self, input_size, output_size):
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    #Prediction function
    def forward(self, x):
        out = self.linear(x)
        return out

# create the linear regression model, print out the parameters
lr = LR(1, 1)
print("The parameters: ", list(lr.parameters()))
print("Linear model: ", lr.linear)   

#try our customize linear regression model with single input
x = torch.tensor([[1.]])
yhat = lr(x)
print("The prediction: ", yhat)






