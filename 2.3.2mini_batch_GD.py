import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

# class for plotting the diagrams

class plot_error_surfaces(object):
    
    # Constructor
    def __init__(self, w_range, b_range, X, Y, n_samples = 30, go = True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)    
        Z = np.zeros((30, 30))
        count1 = 0
        self.y = Y.numpy()
        self.x = X.numpy()
        for w1, b1 in zip(w, b):
            count2 = 0
            for w2, b2 in zip(w1, b1):
                Z[count1, count2] = np.mean((self.y - w2 * self.x + b2) ** 2)
                count2 += 1
            count1 += 1
        self.Z = Z
        self.w = w
        self.b = b
        self.W = []
        self.B = []
        self.LOSS = []
        self.n = 0
        if go == True:
            plt.figure()
            plt.figure(figsize = (7.5, 5))
            plt.axes(projection = '3d').plot_surface(self.w, self.b, self.Z, rstride = 1, cstride = 1, cmap = 'viridis', edgecolor = 'none')
            plt.title('Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
            plt.figure()
            plt.title('Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.show()
            
     # Setter
    def set_para_loss(self, W, B, loss):
        self.n = self.n + 1
        self.W.append(W)
        self.B.append(B)
        self.LOSS.append(loss)
    
    # Plot diagram
    def final_plot(self): 
        ax = plt.axes(projection = '3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W, self.B, self.LOSS, c = 'r', marker = 'x', s = 200, alpha = 1)
        plt.figure()
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
    
    # Plot diagram
    def plot_ps(self):
        plt.subplot(121)
        plt.ylim()
        plt.plot(self.x, self.y, 'ro', label = "training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label = "estimated line")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Data Space Iteration: '+ str(self.n))
        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.title('Loss Surface Contour')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()


""" MAKE SOME DATA """

import torch 
torch.manual_seed(1)

#generate the data with noise and the line
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = 1 * X -1
Y = f + 0.1 * torch.randn(X.size())

#plot the line and the data

# plt.plot(X.numpy(), Y.numpy(), 'rx', label = 'y')
# plt.plot(X.numpy(), f.numpy(), label ='f')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()

""" CREATE THE MODEL AND COST FUNCTION (TOTAL LOSS) """
#define the prediction function
def forward(x):
    return w * x + b 

# define the cost or criterion function
def criterion(yhat, y):
    return torch.mean((yhat-y) ** 2)

# create a plot_error_surfaces object
get_surface = plot_error_surfaces(15, 13, X, Y, 30)

""" BGD """
w = torch.tensor(-15.0, requires_grad=True)
b = torch.tensor(-10., requires_grad=True)
lr = 0.1
LOSS = []
def train_BGD(iter):
    #loop
    for i in range(iter):
        # make prediction 
        Yhat = forward(X)
        # calculate loss
        loss = criterion(Yhat, Y)
        LOSS.append(loss)
        loss.backward()

        #update w, b
        w.data = w.data - lr * w.grad.data
        b.data = b.data - lr * b.grad.data

        # delete gradients
        w.grad.data.zero_()
        b.grad.data.zero_()



train_BGD(10)


""" MINI BATCH GD: BATCH SIZE EQUALS 10 """
from torch.utils.data import Dataset, DataLoader
#create class Data
class Data(Dataset):
    #constructor
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.y = 1 * X - 1
        self.len = self.x.shape[0]
    
    #getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    #get length
    def __len__(self):
        return self.len
    

#create data Object and dataloader object
dataset = Data()
trainLoader = DataLoader(dataset = dataset, batch_size = 5)
print(trainLoader)

#define train_M_BGD
w = torch.tensor(-15., requires_grad=True)
b = torch.tensor(-10., requires_grad=True)

LOSS_MINI = []
lr = 0.1

def train_MBGD(epochs):
    #loop
    for epoch in range(epochs):
        Yhat = forward(X)
        LOSS_MINI.append(criterion(Yhat, Y).tolist())
        for x, y in trainLoader:
            yhat = forward(x)
            loss = criterion(yhat, y)
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            w.grad.data.zero_()


train_MBGD(10)

""" PRACTICE WITH BATCH SIZE OF 20 """
dataset = Data()
trainloader20 = DataLoader(dataset = dataset, batch_size = 20)

w = torch.tensor(-15., requires_grad=True)
b = torch.tensor(-10., requires_grad=True)
LOSS_20 = []
lr = 0.1

def train_M_BGD20(epochs):
    for epoch in range(epochs):
        Yhat = forward(X)
        LOSS_20.append(criterion(Yhat, Y).tolist())

        for x, y in trainloader20:
            #make prediction
            yhat = forward(x)
            #calculate loss
            loss = criterion(yhat, y)
            loss.backward()
            #update w, b
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data

            w.grad.data.zero_()
            b.grad.data.zero_()

train_M_BGD20(10)




#plot out the loss for earch method

plt.plot(LOSS_MINI, label = 'MBGD_5')
plt.plot(LOSS, label = 'MBGD_20')
plt.plot(LOSS_20, label= 'MB')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
        