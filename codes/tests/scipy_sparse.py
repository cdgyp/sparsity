import torch
import numpy as np
from scipy import sparse
import time

size = 50000
density = 0.0001
device = 'cuda'
print('SIZE:', size, 'DENSITY:', density, 'DEVICE:', device)

A = sparse.rand(size, size, format='coo', density=density).astype(np.float32)
b = torch.rand(size, 1, device=device)

values = A.data
indices = np.vstack((A.row, A.col))

i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = A.shape

A_torch = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)

# s = time.time()
# A_torch.mm(b)
# t_torch = time.time() - s
# print('torch: {:g} seconds'.format(t_torch))

b = b.cpu().numpy()

s = time.time()
A.dot(b)
t_np = time.time() - s
print('np:    {:g} seconds'.format(t_np))

# print('torch/np: {:g}'.format(t_torch / t_np))
# print('-'*40)