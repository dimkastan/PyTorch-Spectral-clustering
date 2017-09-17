"""
  Implementing various dimensionality reduction methods with PyTorch Tensors

  
  Under development. Please use with caution.

"""
import torch
from torch.autograd import Variable


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

 
 
def distance_matrix(mat):
    d= ((mat.unsqueeze (0)-mat.unsqueeze (1))**2).sum (2)**0.5
    return d
 
def similarity_matrix(mat):
    d= distance_matrix(mat)
    D = torch.exp(-(d ))
    return D.sqrt()
 
# Generate Clusters
mat = torch.cat([torch.randn(500,2)+torch.Tensor([-2,-3]),   torch.randn(500,2)+torch.Tensor([2,1])])
 
mat = mat[torch.randperm(mat.size(0))]
plt.scatter(mat[:,0].numpy(),mat[:,1].numpy())
plt.show(block=False)

##-------------------------------------------
#         Spectral analysis on distance matrix
##-------------------------------------------
d= distance_matrix(mat);

plt.figure()
plt.imshow(d.numpy())
plt.title('Distance Matrix-Before Ordering')
plt.show(block=False)


[u,s,v]=torch.svd(d)

colors = cm.rainbow(np.linspace(0, 1, mat.size(0)))
[val, ind] = torch.sort(u[:,1] )
plt.figure()

sorted_u  = u[ind,:]
 
for x, color in zip(sorted_u.numpy(), colors):
    plt.scatter(x[1],x[2], color=color)

plt.title('Eigenvector-Mapping')
plt.show(block=False)


plt.figure()
plt.imshow(d[[ind]][:,ind].numpy())
plt.show(block=False)
plt.title('Sorted Matrix');

plt.figure()
plt.plot(torch.sort(u[:,1 ])[0].numpy())
plt.show(block=False)
plt.title("Sorted Eigenvector")

raw_input("Press Enter to exit..") 
plt.close('all')

 
 
