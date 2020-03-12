import torch
from torch.utils.data import Dataset 
torch.manual_seed(1)
from torchvision import transforms
""" SIMPLE DATASET    """
#try to create our awn dataset class

#define class for dataset

class toy_set(Dataset):
    
    # Constructor with defult values 
    def __init__(self, length = 100, transform = None):
        self.len = length
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform
     
    # Getter
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)     
        return sample
    
    # Get Length
    def __len__(self):
        return self.len


#create Dataset Object. Find out the value on index 1. Find out
# the length of Dataset Object.

our_dataset = toy_set()
print('----------',our_dataset[1])
#print("our toy_set object: ", our_dataset)
#print("Value on index 0 of our toy_set object: ", our_dataset[0])
#print("Our toy_set length: ", len(our_dataset))


# Use loop to print out first 3 elements in dataset

for i in range(3):
    x, y = our_dataset[i]
    print("index: ", i, ";x:", x, ";y:", y)


""" PRACTICE """
#try to create a new object with length 50, print out the length of the object

# class my_object(Dataset):

#     #constructor 
#     def __init__(self, length = 50, transform = None):
#         self.len = length
#         self.x = 2 * torch.ones(length, 100)
#         self.y = torch.ones(length, 200)
    
#     #getter
#     def __getitem__(self, index):
#         sample = self.x[index], self.y[index]
#         return sample
#     def __len__(self):
#         return self.len
    

# test = my_object()
# print("my object length: ", len(test))

my_object = toy_set(length = 50)

print("my object length: ",len(my_object))


""" TRANSFROMS """
# create transform class add_
class add_mult(object):

    #constructor
    def __init__(self, addx = 1, muly = 2):
        self.addx = addx
        self.muly = muly

    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.addx
        y = y * self.muly
        sample = x, y
        return sample
    
## create an add_mul transform object and an toy_set object
a_m = add_mult()
data_set = toy_set()

# use loop to print out first 10 elements in dataset

for i in range(10):
    x, y = data_set[i]
    print("index: ", i, "Original x: ", x, "Original y: ", y)
    x_, y_ = a_m(data_set[i])
    print("index: ", i, "Transformed x_: ", x_, "Transformed y_: ", y_)


""" PRACTICE """

class my_add_mult(object):

    #constructor
    def __init__(self, addx = 2, muly = 10):
        self.addx = addx
        self.muly = muly
    
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]

        x = x + self.addx
        y = y + self.addx

        x = x * self.muly
        y = y * self.muly

        return x, y
        
## Test
my_data_set = toy_set(transform=my_add_mult())

for i in range(3):
    x_, y_ = my_data_set[i]
    print("index: ", i, "transformed x: ", x_, "Transform y: ", y_)
        

""" COMPOSE """

class mult(object):

    #constructor
    def __init__(self, mult = 100):
        self.mult = mult
    
    #executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mult
        y = y * self.mult
        sample = x, y
        return sample

#combine the add_mult() and mult()

data_transform = transforms.Compose([add_mult(), mult()])
print("The combination of transform (Compose): ", data_transform)

compose_data_set = toy_set(transform=data_transform)

for i in range(3):
    print("The 3 first elements of Compose_data_set: ", compose_data_set[i])



""" PRACTICE """

# make a compose as mult() exucute first then add_mult

my_compose = transforms.Compose([mult(), add_mult()])

my_compose_data_set = toy_set(transform=my_compose)
# test
for i in range(3):
    print("My data set: ", my_compose_data_set[i])