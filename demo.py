"""
  Implementing various dimensionality reduction methods with PyTorch Tensors

  Diffusion Maps, Laplacian EigenMaps, stc


  Under development. Please use with caution.

"""
import torch
from torch.autograd import Variable


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# adopted from Fransisco Massa's comment
# (x - y)^2 = x^2 - 2*x*y + y^2
def distance_matrix(mat):
    # get the product x * y
    # here, y = x.t()
    r = torch.mm(mat, mat.t())
    # get the diagonal elements
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    # compute the distance matrix
    D = diag + diag.t() - 2*r
    return D.sqrt()

# (x - y)^2 = x^2 - 2*x*y + y^2
def similarity_matrix(mat):
    # get the product x * y
    # here, y = x.t()
    r = torch.mm(mat, mat.t())
    # get the diagonal elements
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    # compute the distance matrix
    D = torch.exp(-(diag + diag.t() - 2*r) )
    return D.sqrt()

# (x - y)^2 = x^2 - 2*x*y + y^2
def diffusion_distance(mat, sigma=8, alpha=0.5):
    D =distance_matrix(mat);
    K = torch.exp(-(torch.pow(torch.div(D,sigma) ,2))) # Kernel
    p = K.sum(1)
    K1 = K/(torch.pow(p.unsqueeze(1)*p,alpha)) # alpha = 1 Laplace Beltrami, 0.5 Fokker Planck diffusion.
    v = torch.sqrt(K1.sum(1))
    A = K1/(v.unsqueeze(1)*v)
    [u,s,v]=torch.svd(D)
    u=u/u[:,0].unsqueeze(1)    
    return K1,u
 
# Generate Clusters
mat = torch.cat([torch.randn(500,2)+torch.Tensor([-2,-3]),   torch.randn(500,2)+torch.Tensor([2,1])])
 
mat = mat[torch.randperm(mat.size(0))]
plt.scatter(mat[:,0].numpy(),mat[:,1].numpy())
plt.show()

##-------------------------------------------
#         Spectral analysis on distance matrix
##-------------------------------------------
d= distance_matrix(mat);

plt.figure(1)
plt.imshow(d.numpy())
plt.title('Distance Matrix-Before Ordering')
plt.show(block=False)


[u,s,v]=torch.svd(d)

colors = cm.rainbow(np.linspace(0, 1, mat.size(0)))
[val, ind] = torch.sort(u[:,1] )
plt.figure(2)

sorted_u  = u[ind,:]
 
for x, color in zip(sorted_u.numpy(), colors):
    plt.scatter(x[1],x[2], color=color)

plt.title('Eigenvector-Mapping')
plt.show(block=False)


plt.figure(3)
plt.imshow(d[[ind]][:,ind].numpy())
plt.show(block=False)
plt.title('Sorted Matrix');

plt.figure(4)
plt.plot(torch.sort(u[:,1 ])[0].numpy())
plt.show(block=True)
plt.title("Sorted Eigenvector")

 
 
