import torch
from torch.autograd import Variable

xy = Variable(torch.FloatTensor([5, 3]), requires_grad = True)

print(xy)