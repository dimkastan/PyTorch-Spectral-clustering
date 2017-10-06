# PyTorch-Spectral-clustering
[Under development]- Implementation of various methods for dimensionality reduction and spectral clustering with PyTorch and Matlab equivalent code.
<br />
Sample Images from PyTorch code
<br />
<p float="left">
<img src="PytorchInputData.png" alt="Input Data" title="Input Data" width="240" height="240"/>
<img src="PytorchFiedlerVector.png" alt="Fiedler Vector" title="Fiedler Vector" width="240" height="240"/>
<img src="Pytorchclusters.png" alt="Clusters" title="Clusters" width="240" height="240"/>
</p>

<br />
Drawing the second eigenvector on data (diffusion map)
<img src="EigenvectorOnData.png" alt="Diffusion Map- Second Eigenvector on data" title="Diffusion Map, Second Eigenvector" width="240" height="240"/>
<br>
Drawing the point-wise diffusion distances 
 <img src="PointDistance.png" alt="Diffusion Map- point-wise distances" title="Diffusion Map, distances" width="240" height="240"/>

<br>
Sorting matrix
<br>
 <p float="left">
<img src="DistanceMatrixBeforeSorting.png" alt="Unsorted PairWiseDistance Matrix" title="Unsorted PairWiseDistance Matrix" width="360" height="300"/>
<img src="Sorted_matrix.png" alt="Sorted Distance Matrix" title="Sorted Distance Matrix" width="360" height="300"/>
 
</p>

<br/><br/> 
 
<br />
## Goal
Use with Pytorch for general purpose computations by implementing some very elegant methods for dimensionality reduction and graph spectral clustering. 
<br />

## Description
In this repo, I am using PyTorch in order to implement various methods for dimensionality reduction and spectral clustering.
At the moment, I have added Diffusion Maps [1] and I am working on the methods presented in the following list (as well as some other that I will add in the future).  
<br />

Except from some examples based on 2-D Gaussian distributed clusters I will also add examples with face, food, imagenet categories etc.
<br />


## Prerequisites
In order to run these examples you need to have Pytorch installed in your system. I worked with Anaconda2 and Pytorch:<br />

    pytorch                   0.2.0           py27hc03bea1_4cu80  [cuda80]  soumith
<br />
(you can verify your pytorch installation by running 

    conda list | grep pytorch

Feel free to contact me for suggestions, comments etc.

### References
 - [1]  Diffusion maps, RR Coifman, S Lafon, Applied and computational harmonic analysis 21 (1), 5-30 <br /> 
 - [2]  Jianbo Shi and Jitendra Malik (1997): "Normalized Cuts and Image Segmentation", IEEE Conference on Computer Vision and Pattern Recognition, pp 731–737 <br />
 - [3] Andrew Y. Ng, Michael I. Jordan, and Yair Weiss. 2001. On spectral clustering: analysis and an algorithm. In Proceedings of the 14th International Conference on Neural Information Processing Systems: Natural and Synthetic (NIPS'01), T. G. Dietterich, S. Becker, and Z. Ghahramani (Eds.). MIT Press, Cambridge, MA, USA, 849-856. 
 - [4] ...
 
