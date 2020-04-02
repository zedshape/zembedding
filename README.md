# Z-Embedding
**Z-Embedding** is a novel algorithm for solving the classification and clustering problem of event intervals with fast speed.  Here we offer our new algorithm **Z-Embedding** with clustering and classification code we used in the paper submission and future research! This README describes how to run the code and how to repeat our experimental research in the paper.

## Dependencies
**Z-Embedding**  has the following dependencies. In our paper, all the libraries and codes are run on Python 3.7 environment. Here are the libraries and their versions we used in our experiments. It is highly recommended to use the same version to get the same result with the experiments on the paper.

**Internal libraries**
These libraries are used in the code, but as they are built-in, you don't need to install them separately. These are not relevant to the algorithm itself, but are used as helper functions.
> `csv`: used to load the datasets
> `time`: used to check the runtime
> `collection.Counter`: used to calculate the purity value

**External libraries**
These are external libraries and you need to install them before importing **Z-Embedding** into your environment. We tried to use the most recent versions at the time of performing the experiments. Short description of usage of each algorithm is also described as follows:
> `numpy=1.18.1`: used to handle general vector-wise calculation
> `scipy=1.4.1`: used to handle sparse matrix
> `sklearn=0.22.1`: used to apply classification (Random forest, 1-NN, SVM) and clustering (K-means) algorithms
> `sklearn_extra=0.0.3`: used to apply extra clustering method (K-medoids)
> `sknetwork=0.12.1`: used to apply SVD calculation for spectral embedding


## Importing Z-Embedding
All the functions needed for **Z-Embedding** is in the single file ZEmbedding.py. Therefore, we can simply import the file in the python shell or in other files as follows:
>  `from ZEmbedding import *` 

Note that ZEmbedding.py is importing util.py inside it, thus you should download the file and put it together with ZEmbedding.py in the same directory. The file util.py contains utility classes and functions used in  **Z-Embedding** for maintaining data structure and for pre-processing.



## Experiments
In the repository, we have some files used to our experiments in the paper as follows:
- Real-world datasets: We used eight real-world datasets and these are available in `data` directory. Detailed information about each dataset is available in our paper. 
 - Available real-world datasets: `BLOCKS.txt`, `HEPATITIS.txt`, `CONTEXT.txt`, `PIONEER.txt`, `SKATING.txt`
	 - Note that you should keep the class labels as well in the same directory: `BLOCKS_CLASSES.txt`, `HEPATITIS_CLASSES.txt`, `CONTEXT_CLASSES.txt`, `PIONEER_CLASSES.txt`, `SKATING_CLASSES.txt`


### Creating ZEmbedding instance
