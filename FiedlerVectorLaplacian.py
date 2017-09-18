"""
% -------------------------------------------------------------
%                                          Matlab code
% -------------------------------------------------------------
% grpah partition using the eigenvector corresponding to the second
% smallest  eigenvalue 
t=[randn(500,2)-repmat([2,2],500,1) ;randn(500,2)+repmat([2,2],500,1)];
scatter(t(:,1),t(:,2))

W=squareform(pdist(t));
A=W<3;
D=sum(A,1);
L= diag(D)-A;
Lsym = diag(D.^-0.5)*L*diag(D.^-0.5);
[u,s,v]=svd(Lsym);
f=find(diag(s)>0.01);
plot(u(:,f(end)))
F=u(:,f(end));
plot(F);title('Second smallest non-zero eigenvalue eigenvector');
scatter(t(F<0,1),t(F<0,2),'b*');hold on
scatter(t(F>0,1),t(F>0,2),'g*');
"""
# Pytorch equivalent code
...
