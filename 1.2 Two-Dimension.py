import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import pandas as pd

#convert 2d LIST to 2D Tensor

twoD_list = [[1,2,3], [4, 5, 6], [7, 8, 9]]
twoD_tensor = torch.tensor(twoD_list)
print("The new 2D Tensor: ", twoD_tensor)

#try .ndimension(), .shape, .size()

print("The dimension of twoDtensor: ", twoD_tensor.ndimension())
print("The size of tensor: ", twoD_tensor.size())
print("The shape of tensor: ", twoD_tensor.shape)


#convert tensor to numpy array,  and back

twoNp = twoD_tensor.numpy()
print("Tensor to numpy array")
print("The numpy array after converting: ", twoNp)
print("Type of numpy: ", twoNp.dtype)
print("=========================")

new_2tensor = torch.from_numpy(twoNp)
print("numpy to tensor")
print("The new tensor: ", new_2tensor)
print("Type after converting: ", new_2tensor.dtype)


#try to convert the panda dataFrame to tensor
df = pd.DataFrame({'a': [1,2,3], 'b': [4, 5, 6]})

print("pandas dataframe: ", df)
print("pandas dataframe to numpy: ", df.values)
print("Type before converting: ", df.values.dtype)

# PRACTICE
#try to convert pandas series to tensor
df = pd.DataFrame({'A': [11, 33, 22], 'B': [3, 3, 2]})
newtensor = torch.from_numpy(df.values)
print("pandas dataframe: ", df, "Type of pd: ", df.values.dtype)
print("tensor from dataFrame: ", newtensor)
print("Type of tensor: ", newtensor.dtype)


# INDEXING AND SLICING

newtensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("the value on 1st-row first two columns", newtensor[0][0:2])
print("the value on 1st-row first two columns", newtensor[0, 0:2])
print("the value on 1st-col first two rows", newtensor[0:2, 0])


## SLICING

slicingTensor = newtensor[1:3]
print("tensor after slicing: ", slicingTensor)
print("Dimension of tensor: ", slicingTensor.ndimension())


# PRACTICE
# use slice and index to change the values on matrix tensor

tensor_ = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
print("Tensor: ", tensor_)
tensor_[1:3, 1] = 0
print("after changing: ", tensor_)


# TENSOR OPERATIONS

#matrix multiplication

A = torch.tensor([[1,2,3], [3, 2, 1], [1, 1, 1]])
B = torch.tensor([[1, 1, 1], [4, 3, 2], [4, 4, 5]])

A__MM__B = torch.mm(A, B)
print("The result of A * B: ", A__MM__B)

A = torch.tensor([[1,2,3], [3, 2, 1]])
B = torch.tensor([[1, 1, 1], [4, 3, 2], [4, 4, 5], [1, 1, 1]])

A__MM__B = torch.mm(A, B)
print("The result of A * B: ", A__MM__B)