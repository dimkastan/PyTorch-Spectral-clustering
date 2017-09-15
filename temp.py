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
def diffusion_distance(mat):
    sigma = 16;
    # get the product x * y
    # here, y = x.t()
    r = torch.mm(mat, mat.t())
    # get the diagonal elements
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    # compute the distance matrix
    K = torch.exp(-(torch.pow((diag + diag.t()- 2*r)/sigma ,2))) 
    p = K.sum(1)
    K1 = K/(torch.pow(p*p.unsqueeze(1),0.5)) # alpha=>1 approx. Laplace–Beltrami operator, 0.5 approximates Fokker-Planck diffusion.
    v = torch.sqrt(K1.sum(1))
    A = K1/(v*v.unsqueeze(1))
    return A

    # D = L2_distance(X',X',1);
# K = exp(-(D/sigmaK).^2);
# p = sum(K);
# p = p(:);
# K1 = K./((p*p').^alpha);
# v = sqrt(sum(K1));
# v = v(:);
# A = K1./(v*v');


mat = torch.cat([torch.randn(500,2)+torch.Tensor([-2,-3]),   torch.randn(500,2)+torch.Tensor([3,-2])])

mat = mat[torch.randperm(mat.size(0))]
plt.scatter(mat[:,0].numpy(),mat[:,1].numpy())
plt.show()

##-------------------------------------------
#         Spectral analysis on similarity matrix
##-------------------------------------------
d= similarity_matrix(mat);

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

plt.figure(3)
plt.plot(torch.sort(u[:,1 ])[0].numpy())
plt.show(block=False)
plt.title("Sorted Eigenvector")

 

##-------------------------------------------
#          Diffusion map
##-------------------------------------------
d= diffusion_distance(mat)
dmat= distance_matrix(mat)


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
plt.imshow(dmat[[ind]][:,ind].numpy())
plt.show(block=False)
plt.title('Sorted Matrix');

plt.figure(3)
plt.plot(torch.sort(u[:,1 ])[0].numpy())
plt.show(block=False)
plt.title("Sorted Eigenvector")
