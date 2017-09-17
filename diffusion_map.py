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


color_map =  plt.get_cmap('jet')

 
def distance_matrix(mat):
    d= ((mat.unsqueeze (0)-mat.unsqueeze (1))**2).sum (2)**0.5
    return d

 
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
[d,u,s]= diffusion_distance(mat,4.0,0.5)
plt.figure(1)
plt.imshow(d.numpy(),cmap= color_map)
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
plt.imshow(d[[ind]][:,ind].numpy(),cmap= color_map)
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

assert min_d ==0 , "Error in distance matrix"
 

random_point = min(torch.round(torch.abs(torch.randn(1)/2.0)*len(mat))[0],len(mat));

values = u[:,1 ] 
norm  = colors.Normalize(vmin=values.min(), vmax=values.max())
for t in range(0,10,1):
    plt.figure();
    values = u[:,1 ]*(s[1]**t) 
    scalarMap = cm.ScalarMappable( norm=norm , cmap=color_map)
    plt.scatter(mat[:,0].numpy(),mat[:,1].numpy())
    for i in range(len(values)):
        color = scalarMap.to_rgba(values[i])
        plt.scatter(mat[i,0],mat[i,1], color=color)

    plt.show(block=False)
    plt.title("Second Eigenvector at time:"+str(t))
    plt.pause(0.1) 
    p = torch.pow(s,t)
    data =   u[:,1:3]*(p[1:3].expand_as(u[:,1:3]))
    d=distance_matrix(data)
    plt.figure();
    plt.imshow(d[[ind]][:,ind].numpy(),cmap= color_map, vmin= 0, vmax=max_d)
    plt.title('distance matrix at time:'+str(t))    
    plt.show(block=False)
    # draw the distances from one point
    plt.figure()
    plt.scatter(mat[:,0].numpy(),mat[:,1].numpy())
    for i in range(len(data)):
        color = scalarMap.to_rgba(d[int(random_point),i]) # take the distance from one point
        plt.scatter(mat[i,0],mat[i,1], color=color)

    plt.scatter(mat[int(random_point),0],mat[int(random_point),1], color=[0.0 ,0.0,0.0], marker="*")  
    plt.title('distance from point at time:'+str(t))
    plt.show(block=False)
    raw_input("Press Enter to continue..") 


raw_input("Press Enter to exit..") 
plt.close('all')
