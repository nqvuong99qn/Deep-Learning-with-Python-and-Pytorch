import numpy as np 
import matplotlib.pyplot as plt 

# The class for plotting

class plot_diagram():

    #constructor
    def __init__(self, X, Y, w, stop, go =False):
        start = w.data 
        self.error = []
        self.parameter = []
        self.X = X.numpy()
        self.Y = Y.numpy()
        self.parameter_values = torch.arange(start, stop)
        self.Loss_function = [criterion(forward(X), Y) for w.data in self.parameter_values]
        w.data = start

    #executor
    def __call__(self, Yhat, w, error, n):
        self.error.append(error)
        self.parameter.append(w.data)
        plt.subplot(212)
        plt.plot(self.X, Yhat.detach().numpy())
        plt.plot(self.X, self.Y, 'ro')
        plt.xlabel("A")
        plt.ylim(-20, 20)
        plt.subplot(211)
        plt.title("data space Estimated Line" + str(n))
        plt.plot(self.parameter_values.numpy(), self.Loss_function)
        plt.plot(self.parameter, self.error, 'ro')
        plt.xlabel("B")
        plt.figure()

    #destructor
    def __del__(self):
        pass

""" MAKE SOME DATA """
#import the library pytorch
import torch
#create  the f(X) with a slope of -3
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = 3 * X
# plot this line
plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
#plt.show()

Y = f + 0.1 * torch.randn(X.size())

#Plot the data point (new line with noise)
plt.plot(X.numpy(), Y.numpy(), 'rx', label = 'Y')
plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
#plt.show()
        
""" CREATE THE MODEL AND COST FUNCTION (tOTAL LOSS) """
#create forward function for prediction

def forward(x):
    return w * x

#create the MSE function for evaluate the reslut
def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)

# create learning rate and an empty list to record the loss
# for each iteration
lr = 0.1
LOSS = []

w = torch.tensor(-10.0, requires_grad=True)
gradient_plot = plot_diagram(X, Y, w, stop = 5)

# define a function for train the model
def train_mode(iter):
    for epoch in range(iter):
        #make the prediction
        Yhat = forward(X)
        # calculate loss
        loss = criterion(Yhat, Y)
        #plot the diagram for us to have a better idea
        gradient_plot(Yhat, w, loss.item(), epoch)

        #store the loss into list
        LOSS.append(loss)

        #backward pass: compute gradient of the loss with 
        loss.backward()
        #update parameters
        w.data = w.data - lr * w.grad.data
        # zero the gradients before running the backward pass
        w.grad.data.zero_()

train_mode(4)

#plot the loss for each iteration

plt.plot(LOSS)
plt.tight_layout()
plt.xlabel('Epoch/Iterations')
plt.ylabel("Cost")
#plt.show()

""" PRACTICE """
#practice: create w with the initial value of -15.0
w = torch.tensor(-15., requires_grad=True)
#create LOSS2 LIST
LOSS2 = []
#create  my_train_model
gradient_plot1 = plot_diagram(X, Y, w, stop = 15)

def my_train_model(iter):
    for epoch in range(iter):
        # make prediction
        Yhat = forward(X)
        #calculate the iteration
        loss = criterion(Yhat, Y)
        #plot the diagram to have a better idea
        gradient_plot1(Yhat, w, loss.item(), epoch)

        #store the loss into list
        LOSS2.append(loss)
        # backward pass: compute gradient of the loss
        loss.backward()
        # update parameter
        w.data = w.data - lr * w.grad.data

        #zero the gradients before running the backward
        w.grad.data.zero_()

my_train_model(15)
plt.plot(LOSS2)
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")

plt.show()



