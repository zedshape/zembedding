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
All the functions needed for **Z-Embedding** is in the single file `ZEmbedding.py`. Therefore, we can simply import the file in the python shell or in other files as follows:
>  `from ZEmbedding import *` 

Note that `ZEmbedding.py` is importing `util.py` inside it, thus you should download the file and put it together with `ZEmbedding.py` in the same directory. The file `util.py` contains utility classes and functions used in  **Z-Embedding** for maintaining data structure and for pre-processing.



## Experiments
In the repository, we have some files used to our experiments in the paper as follows:
- Real-world datasets: We used eight real-world datasets and these are available in `data` directory. Detailed information about each dataset is available in our paper. 
 - Available real-world datasets: `BLOCKS.txt`, `HEPATITIS.txt`, `CONTEXT.txt`, `PIONEER.txt`, `SKATING.txt`
	 - Note that you should keep the class labels as well in the same directory without changing their names: `BLOCKS_CLASSES.txt`, `HEPATITIS_CLASSES.txt`, `CONTEXT_CLASSES.txt`, `PIONEER_CLASSES.txt`, `SKATING_CLASSES.txt`. 


### Creating Z-Embedding instance
After importing `ZEmbedding.py`, you are ready to create **Z-Embedding** object. This object creates spectral embedding space for the dataset, that can be used for classification and clustering problem. Thus, whenever you use **Z-Embedding**, this step will be the first step you should run. The basic usage is as follows:
>  `object = createZEmbedding(filename, dim, minSup, maxSup, gap, printing)` 

We receive the filename and four parameters to create embedding space. Detailed explanation for each parameter is available in the paper. For example, if you want to create an embedding space for `BLOCKS` dataset, you can create the instance as follows:
>  `blocks = createZEmbedding('data/BLOCKS.txt', 4)` 

if you don't enter the three constraints `{minsup, maxSup, gap}` like above, no constraint will be applied. The parameters that our experiment used are described in the paper. Default dimension is set to 8. `Printing` option is available if you want to print detailed procedure of the code. There three parameters are relative values with the range of [0, 1]. One example is as follows:
>  `blocks = createZEmbedding('data/BLOCKS.txt', 4, minSup=0.1, maxSup=0.9, gap=0.2)` 

### Clustering
For clustering, you can run our function `NTrialClustering` using the **Z-Embedding** instance created above. This function performs the clustering `n` times and returns average purity and runtime values. This function runs both K-means clustering and K-medoids clustering. All the algorithms are performed using `scikit-learn` library (`sklearn_extra` in the case of K-medoids) with default parameter settings. It receives three parameters:

>  `meanPurity, runtime = NTrialClustering(object, k, n)` 

You can put the **Z-Embedding** instance and `k` meaning the number of clusters, `n` as number of trials. Default number of trials is set to 100. The ground-truth number of clusters for each dataset can be found in the `CLASSES` data files in `data` directory.

### Classification

For classification, you can run our function for n-fold cross validation. It is also required to create **Z-Embedding** instance prior to running classification. We support three classification methods: `SVM, Random Forests, k-NN`. Each algorithm can be run using the same function as follows:

>  `NFoldClassification(object, n, k, cl, kernel, printing=False)`

There are one global parameters applied to every classifier we support: `n` as number of folds. The default number of `n` is set to 10, meaning that the function performs **10-fold cross validation**.

#### Classifiers
If you put no information about the classifier, then the basic option will be `1-NN`. Or, you can state the number of `k` making it `k-NN`.
>  `NFoldClassification(object, k=1)`

For `SVM`, you can state  the algorithm name as `SVM`. 

>  `NFoldClassification(object, n=10, cl='svm')`
>  
The default kernel we set is `rbf`, but you can state other kernels that are available in `scikit-learn`, for example, `poly`.

>  `NFoldClassification(object, n=10, cl='svm', kernel='poly')`

Finally, `Random Forests` are available by putting `cl='rf'`.

>  `NFoldClassification(object, n=10, cl='rf')`

Note that this wrapper functions `NTrialClustering` and `NFoldClassification` are developed for research purposes and to help repeat the experiment. If you want to use more diverse classifiers or to change more parameters, you can directly use **Z-Embedding** object and embedding space and regard it as feature vectors, and apply any of available machine learning algorithms on it. 
