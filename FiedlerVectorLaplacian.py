"""
% -------------------------------------------------------------
%                                          Matlab code
% -------------------------------------------------------------
% grpah partition using the eigenvector corresponding to the second
% smallest  eigenvalue 
% grpah partition using the eigenvector corresponding to the second
% smallest  eigenvalue 
t=[randn(500,2)+repmat([-2,-2],500,1) ;randn(500,2)+repmat([2,2],500,1)];
scatter(t(:,1),t(:,2))
W=squareform(pdist(t));
A=W<3;      % create adjacency matrix (set connected notes equal to one)
D       = sum(A,1);
L       = diag(D)-A;
Lsym    = diag(D.^-0.5)*L*diag(D.^-0.5);
[u,s,v] = svd(Lsym);

figure; plot(u(:, (end-1)))
F       = u(:, (end-1));
plot(F);title('Second smallest non-zero eigenvalue eigenvector');
scatter(t(F<0,1),t(F<0,2),'bo','filled');hold on
scatter(t(F>0,1),t(F>0,2),'go','filled');
"""
# Pytorch equivalent code
import torch
from torch.autograd import Variable


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

 
import matplotlib.colors as colors
import matplotlib.cm as cm 
import matplotlib as mpl


color_map =  plt.get_cmap('jet')
 
def distance_matrix(mat):
    d= ((mat.unsqueeze (0)-mat.unsqueeze (1))**2).sum (2)**0.5
    return d
 

# Generate Clusters
mat = torch.cat([torch.randn(500,2)+torch.Tensor([-2,-3]),   torch.randn(500,2)+torch.Tensor([2,1])]) 
plt.scatter(mat[:,0].numpy(),mat[:,1].numpy())
plt.show(block=False)
##-------------------------------------------
#         Compute distance matrix and then the Laplacian
##-------------------------------------------
d= distance_matrix(mat);
da=d<2;
plt.figure()
plt.imshow(da.numpy())
plt.show(block=False)

D= ((da.float()).sum(1)).diag()
L = D -da.float()
plt.figure()
plt.title("Laplacian")
plt.imshow(L.numpy())
plt.show(block=False)



Lsym=torch.mm(torch.mm(torch.diag(torch.pow(torch.diag(D),-0.5)),L),torch.diag(torch.pow(torch.diag(D),-0.5)));
plt.figure()
plt.imshow(Lsym.numpy())
plt.title("Symmetric Laplacian")
plt.show(block=False)


[u,s,v]=torch.svd(Lsym)

# plot fiedler vector

plt.figure()
plt.title('Fiedler vector')
plt.plot(u[:,-2].numpy());
plt.show(block=False)
norm  = colors.Normalize(vmin=-1, vmax=1)

scalarMap = cm.ScalarMappable( norm=norm , cmap=color_map)


plt.figure()
plt.title('clusters')
for i in range(len(u[:,-2])):
	if u[i,-2]<0:
		color = scalarMap.to_rgba(-1)
		plt.scatter(mat[i,0],mat[i,1], color=color,marker='o')
	else:
		color = scalarMap.to_rgba(1)
		plt.scatter(mat[i,0],mat[i,1], color=color,marker='*')

plt.show(block=False)

raw_input("Press Enter to exit..") 
plt.close('all')
