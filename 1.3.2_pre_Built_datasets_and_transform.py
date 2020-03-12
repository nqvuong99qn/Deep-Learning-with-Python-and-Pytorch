import torch
import matplotlib.pylab as plt
import numpy as np 
torch.manual_seed(0)

# show data by diagram
def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + str(data_sample[1].item()))
    plt.show()

""" PREBUILT DATASETS """

import torchvision.transforms as transforms
import torchvision.datasets as dsets 

dataset = dsets.MNIST(
    root = './data',
    train = False, 
    download= True,
    transform= transforms.ToTensor()
)

## Examinate whether the elements in dataset MNIST are tuples, and what is in the tuple?
print("Type of the first element: ", type(dataset[0]))
print("The length of the tuple: ", len(dataset[0]))
print("The shape of the first element in the tuple: ", dataset[0][0].shape)
print("The type of the first element in the tuple: ", type(dataset[0][0]))
print("The type of the second  element in the tuple: ", type(dataset[0][1]))

# plot the first element in the dataset
#show_data(dataset[0])
# plot the second element int he dataset
show_data(dataset[1])

## Combine two transforms: Crop and convert to tensor, Apply the compose to MNIST dataset 

cropTensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])
dataset = dsets.MNIST(root = './data', train = False, download=True, transform= cropTensor_data_transform)
print("The shape of the first element in the first tuple: ", dataset[0][0].shape)

#show_data(dataset[0], shape=(20, 20))
#show_data(dataset[1], shape=(20, 20))


""" PRACTICE """
RandomVerticalFlip = transforms.Compose([transforms.RandomVerticalFlip(p = 1), transforms.ToTensor()])
dataset = dsets.MNIST(root = './data', train = False, download=True, transform=RandomVerticalFlip)

show_data(dataset[1])

