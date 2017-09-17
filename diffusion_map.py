"""
  Implementing various dimensionality reduction methods with PyTorch Tensors

  Diffusion Maps, Laplacian EigenMaps, stc


  Under development. Please use with caution.

"""
import torch
from torch.autograd import Variable


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm 
import matplotlib as mpl


jet =  plt.get_cmap('jet')


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
def diffusion_distance(mat, sigma=8.0, alpha=1.0):
    D =distance_matrix(mat);
    K = torch.exp(-(torch.pow(torch.div(D,sigma) ,2))) # Kernel
    p = K.sum(1)
    K1 = K/(torch.pow(p.unsqueeze(1)*p,alpha)+1e-9) # alpha = 1 Laplace Beltrami, 0.5 Fokker Planck diffusion.
    v = torch.sqrt(K1.sum(1))
    A = K1/(1e-9+v.unsqueeze(1)*v)
    [u,s,v]=torch.svd(A)
    u=u/(1e-9+u[:,0].unsqueeze(1))
    return K1,u,s

# Generate Clusters
mat = torch.cat([torch.randn(500,2)+torch.Tensor([-2,-3]),   torch.randn(500,2)+torch.Tensor([2,1])])
 
# mat = mat[torch.randperm(mat.size(0))]
plt.scatter(mat[:,0].numpy(),mat[:,1].numpy())
plt.show(block=False)
plt.pause(1)
 

##-------------------------------------------
#          Diffusion map
##-------------------------------------------
[d,u,s]= diffusion_distance(mat,2.0,1.0)
plt.figure(1)
plt.imshow(d.numpy())
plt.title('Distance Matrix-Before Ordering')
plt.show(block=False)

color_vals = cm.rainbow(np.linspace(0, 1, mat.size(0)))
[val, ind] = torch.sort(u[:,1] )
plt.figure(2)

sorted_u  = u[ind,:]
for x, color in zip(sorted_u.numpy(), color_vals):
    plt.scatter(x[1],x[2], color=color)

plt.title('Eigenvector-Mapping')
plt.show(block=False)
plt.pause(0.1)

plt.figure(3)
plt.imshow(d[[ind]][:,ind].numpy())
plt.show(block=False)
plt.title('Sorted Matrix');
plt.pause(0.1)

plt.figure(4)
plt.plot(torch.sort(u[:,1 ])[0].numpy())
plt.show(block=False)
plt.title("Sorted Eigenvector")
plt.pause(0.1)

 

data =   u[:,1:4]*(torch.pow(s[1:4].expand_as(u[:,1:4]),0))
d=distance_matrix(data)
min_d = d.min();
max_d = d.max();

 

values = u[:,1 ] 
norm  = colors.Normalize(vmin=values.min(), vmax=values.max())
for t in range(0,32,4):
    plt.figure();
    values = u[:,1 ]*(s[1]**t) 
    scalarMap = cm.ScalarMappable( norm=norm , cmap=jet)
    plt.scatter(mat[:,0].numpy(),mat[:,1].numpy())
    for i in range(len(values)):
        color = scalarMap.to_rgba(values[i])
        plt.scatter(mat[i,0],mat[i,1], color=color)

    plt.show(block=False)
    plt.title("Second Eigenvector at time:"+str(t))
    plt.pause(0.5)    
    raw_input("Press Enter to continue..") 


raw_input("Press Enter to exit..") 
plt.close('all')
