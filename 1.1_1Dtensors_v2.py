import torch
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

#matplotlib inline 


# plot vectors 
#param: Vectors 

def plotVec(vectors):
    ax = plt.axes()

    #for loop to draw the vectors
    for vec in vectors:
        ax.arrow(0, 0, *vec["vector"], head_width = 0.05, color = vec["color"], head_length = 0.1)
        plt.text(*(vec["vector"] + 0.1), vec["name"])

    plt.ylim(-2, 2)
    plt.xlim(-2, 2)


"""             Types and Shape         """

#convert a integer/float list with length 5 to a tensor

ints_to_tensor = torch.tensor([0, 1, 2, 3, 4])
floasts_to_tensor = torch.tensor([1.2, 3.4, 4.5, 6.7])
print("The dtype of tensor object after converting integer list to tensor: ",ints_to_tensor.dtype)
print("The type of tensor object after converting integer list to tensor: ",ints_to_tensor.type)

print("The dtype of tensor object after converting float list to tensor: ", floasts_to_tensor.dtype)
print("The type of tensor object after converting float list to tensor: ", floasts_to_tensor.type)


# NOTE: THE ELEMENTS IN THE LIST THAT WILL BE CONVERTED TO TENSOR MUST HAVE THE SAME TYPE
# convert a integer list  to float tensor

ins_to_floatTensor = torch.FloatTensor([1, 2, 3, 4])

print("The dtype of tensor object after converting integer list to float tensor: ", ins_to_floatTensor.dtype)
print("The type of tensor object after converting integer list to float tensor: ", ins_to_floatTensor.type)


print("The size of the tensor: ", ins_to_floatTensor.size())
print("The dimension of the tensor: ", ins_to_floatTensor.ndimension())

#using view has been reshaped from one-dimensional tensor object to a two-dimensional tensor object 
newTensor = ins_to_floatTensor.view(4, 1)
print("Original Size: ", ins_to_floatTensor)
print("After changing: ", newTensor)

newTensor1 = ins_to_floatTensor.view(-1, 1)
print("Original Size: ", ins_to_floatTensor)
print("After changing: ", newTensor1)


# NUMPY ARRAY TO A TENSOR

#convert a numpy array to a tensor and drawback

numpy_array = np.array([0., 1., 2., 3. ])
new_tensor = torch.from_numpy(numpy_array)
print("The dtype of new tensor: ", new_tensor.dtype)
print("The type of new tensor: ", new_tensor.type)

back_to_numpy = new_tensor.numpy()
print("The numpy array from tensor: ", back_to_numpy)
print("The dtype of numpy array: ", back_to_numpy.dtype)

""" 
    back_to_numpy and new_tensor still point to numpy_array. As a result if we change numpy_array both
    back_to_numpy and new_tensor will change
"""


# PANDAS AND TENSOR
#convert a panda series to a tensor
pandas_series = pd.Series([0.1, 0.2, 0.3, 0.4])
new_tensor = torch.from_numpy(pandas_series.values)
print("the new tensor from numpy array: ", new_tensor)
print("The dtype of new tensor: ", new_tensor.dtype)
print("The type of new tensor: ", new_tensor.type())

## PRACTICE
your_tensor = torch.tensor([1, 2, 3, 4, 5])
new_tensor = your_tensor.view(5, 1)
print("The new tensor after converting: ",new_tensor)
print("The size of tensor: ", new_tensor.size)
print("The dimension of tensor: ", new_tensor.ndimension())

### INDEXING AND SLICING
#indexing in tensor the same index in array

#slice  new_tensor

subset_tensor_sample = your_tensor[1:4]
print("Original tensor: ", your_tensor)
print("The subset of tensor: ", subset_tensor_sample)

#using variable to contain the selected index, and pass it to slice operation
selected_index = [3, 4]
subset_tensor_sample = your_tensor[selected_index]
print("Original tensor: ", your_tensor)
print("The subset of tensor: ", subset_tensor_sample)

## PRACTICE
# change the values on index 3 4, 7 to 0
practice_tensor = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9])
selected_index = [3, 4, 7]
practice_tensor[selected_index] = 0

print(practice_tensor)


## TENSOR FUNCTION

## MEAN AND STANDARD DEVIATION, MAX, MIN, 

#sample tensor for mathmatic calculation methods on tensor

math_tensor = torch.tensor([1.0, -1., 1, -1])
print("Tensor: ", math_tensor)
mean = math_tensor.mean()
print("the mean of the tensor: ", mean)
standard_deviation = math_tensor.std()
print("The standard deivation of math_tensor: ", standard_deviation)



## CREATE TENSOR BY torch.linspace()

len_5_tensor = torch.linspace(-2, 2, steps = 5)
print("First try on linspace: ", len_5_tensor)


## TENSOR OPERATIONS

#create two sample tensors
u = torch.tensor([1, 0])
v = torch.tensor([0, 1])
w = u + v
#plot u, v, w
plotVec([
    {"vector": u.numpy(), "name": 'u', "color": 'r'},
    {"vector": v.numpy(), "name": 'v', "color": 'b'},
    {"vector": w.numpy(), "name": 'w', "color": 'g'}
])