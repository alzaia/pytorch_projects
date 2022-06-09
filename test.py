import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from matplotlib.colors import to_rgba
from tqdm.notebook import tqdm  # Progress bar

print("Using torch", torch.__version__)
torch.manual_seed(42)  # Setting the seed

x = torch.Tensor(2, 3, 4)
print(x)

# Create a tensor from a (nested) list
x = torch.Tensor([[1, 2], [3, 4]])
print(x)

# Create a tensor with random values between 0 and 1 with the shape [2, 3, 4]
x = torch.rand(2, 3, 4)
print(x)

shape = x.shape
print("Shape:", x.shape)

size = x.size()
print("Size:", size)

dim1, dim2, dim3 = x.size()
print("Size:", dim1, dim2, dim3)

np_arr = np.array([[1, 2], [3, 4]])
tensor = torch.from_numpy(np_arr)

print("Numpy array:", np_arr)
print("PyTorch tensor:", tensor)

tensor = torch.arange(4)
np_arr = tensor.numpy()

print("PyTorch tensor:", tensor)
print("Numpy array:", np_arr)



a = torch.rand(3, 4, 6)
print(a)

b = torch.ones(3, 4, 6)
print(b)

c = 4*a + 3*b
print(c)









