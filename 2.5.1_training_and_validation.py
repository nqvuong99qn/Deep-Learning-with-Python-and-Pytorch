import torch 
from torch import nn, optim
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
#create data class

class Data(Dataset):
    #constructor 
    def __init__(self, train = True):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.f = -3 * self.x + 1
        self.y = self.f + 0.1 * torch.randn(self.x.size())
        self.len = self.x.shape[0]
        #outliers
        if train == True:
            self.y[0] = 0
            self.y[50:55] = 20
        else:
            pass
    #getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    
    # get length
    def __len__(self):
        return self.len 
    
# create training dataest and validation dataset
train_data = Data()
val_data = Data(train = False)

#plot out training point
# plt.plot(train_data.x.numpy(), train_data.y.numpy(), label = 'train')
# plt.plot(train_data.x.numpy(), train_data.f.numpy(), label = 'function')
# plt.legend()
#plt.show()

""" CREATE A LINEAR REGRESSION OBJECT, DATA LOADER, CRITERION FUNCTION """

#create linear regression class
class Linear_Regression(nn.Module):
    #constructor
    def __init__(self, inputsize, outputsize):
        super(Linear_Regression, self).__init__()
        self.linear = nn.Linear(inputsize, outputsize)
    
    #forward
    def forward(self, x):
        yhat = self.linear(x)
        return yhat

#create MSELOSS FUNCTION AND DATALOADER
criterion = nn.MSELoss()
trainLoader = DataLoader(dataset= train_data, batch_size=1)


""" Different learning rates and data Structure to store results for
    different Hyperparameters """

#creat learning rate list, the error lists and the models list
learning_rates = [0.0001, 0.001, 0.01, 0.1]
train_error = torch.zeros(len(learning_rates))

validation_error = torch.zeros(len(learning_rates))

MODELS = []

#define the train model function and train the model

def train_model_with_lr(iter, lr_list):
    for i, lr in enumerate(lr_list):
        model = Linear_Regression(1, 1)
        optimizer = optim.SGD(model.parameters(), lr = lr)
        for epoch in range(iter):
            for x, y in trainLoader:
                yhat = model(x)
                loss = criterion(yhat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()    
            
        #train data
        Yhat = model(train_data.x)
        train_loss = criterion(Yhat, train_data.y)
        train_error[i] = train_loss.item()

        #validation data
        Yhat = model(val_data.x)
        val_loss = criterion(Yhat, val_data.y)
        validation_error[i] = val_loss.item()
        MODELS.append(model)

train_model_with_lr(10, learning_rates)


plt.semilogx(np.array(learning_rates), train_error.numpy(), label = 'training loss / total loss')
plt.semilogx(np.array(learning_rates), validation_error.numpy(), label = 'val loss/total loss')
plt.ylabel('cost/total loss')
plt.xlabel('learning rate')
plt.legend()
#plt.show()


good_model = MODELS[2]
for x, y in trainLoader:
    print('yhat: ', good_model(x), 'y: ', y)




