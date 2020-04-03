from utils import *
# Python built-in
import time
from collections import Counter
# numpy
import numpy as np
# scipy
from scipy import sparse
# scikit-learn
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
# scikit-extra
from sklearn_extra.cluster import KMedoids
# scikit-network
from sknetwork.utils.checks import check_weights
from sknetwork.linalg import SparseLR, LanczosSVD, safe_sparse_dot, diag_pinv
# To prevent showing empty-cluser error
import warnings
warnings.filterwarnings('ignore')

class BiSpectral():
    def __init__(self, embedding_dimension=2, regularization=0.001):
        self.embedding_dimension = embedding_dimension
        self.regularization = regularization

    def returnNormalized(self, adjacency):
        n1, n2 = adjacency.shape
        #total weight heuristic stated in De Lara (2019)
        adjacency = SparseLR(adjacency, [(self.regularization * np.ones(n1), np.ones(n2))])

        #left side of normalized laplacian (squared later)
        w_row = adjacency.dot(np.ones(adjacency.shape[1]))
        #right side of normalized laplacian (squared later)
        w_col = (adjacency.T).dot(np.ones(adjacency.shape[0]))
        self.diag_row = diag_pinv(np.sqrt(w_row))
        self.diag_col = diag_pinv(np.sqrt(w_col))
        normalized_adj = safe_sparse_dot(self.diag_row, safe_sparse_dot(adjacency, self.diag_col))

        return normalized_adj

    def fit(self, adjacency):
        self.solver = LanczosSVD()

        n_components = self.embedding_dimension + 1 # first eigenvector/value is doing nothing
        self.normalized_adj = self.returnNormalized(adjacency)

        # fitting and embedding
        self.solver.fit(self.normalized_adj, n_components)
        index = np.argsort(-self.solver.singular_values_)
        self.singular_values_ = self.solver.singular_values_[index[1:]]
        self.row_embedding_ = self.solver.left_singular_vectors_[:, index[1:]]
        self.col_embedding_ = self.solver.right_singular_vectors_[:, index[1:]]
        self.embedding_ = np.vstack((self.row_embedding_, self.col_embedding_))

        return self

class ZEmbedding:
    def __init__(self, database, constraints, classes):

        self.FL = {}
        self.comparisoncount = 0
        self.totalfrequency = 0
        self.constraints = constraints
        self.database = database
        self.classes = classes

    def pruneWithMinsup(self):
        copiedEvents = self.database.initialSupport.copy()
        # remove event below threshold
        for label, support in self.database.initialSupport.items():
            if support < self.constraints["minSup"] or support > self.constraints["maxSup"]:
                del copiedEvents[label]

        self.database.initialSupport = copiedEvents
        for seq in self.database.sequences:
            prunedSequences = []
            for event in seq.sequences:
                if (
                    event.label in self.database.initialSupport
                    and ((self.database.initialSupport[event.label]
                    >= self.constraints["minSup"] )
                    or (self.database.initialSupport[event.label]
                    <= self.constraints["maxSup"]))
                ):
                    prunedSequences.append(event)
            seq.sequences = prunedSequences
        return

    def createGHashTable(self):
        # each e-sequence id to generate next
        total = 0
        for S in self.database.sequences:
            # iterate every event pairs in the same sequence
            for s1 in S.sequences:
                for s2 in S.sequences:
                    # we keep the order not to make duplication
                    if s1 < s2:
                        R2 = getRelation(s1, s2, self.constraints)
                        if R2 != None:
                            pair = (s1.label, s2.label, R2)
                            # initialization
                            if pair not in self.FL:
                                self.FL[pair] = {S.id: 0}
                            elif S.id not in self.FL[pair]:
                                self.FL[pair][S.id] = 0
                            self.FL[pair][S.id] += 1

        for R2 in list(self.FL):
            if (len(self.FL[R2]) < self.constraints["minSup"]) or (len(self.FL[R2]) > self.constraints["maxSup"]):
                del self.FL[R2]
            else:
                self.totalfrequency += 1

    def constructBipartiteWithWeight(self):
        edgeCount = 0
        edges = {}
        for R2 in self.FL:
            edges[edgeCount] = []
            for sid, count in self.FL[R2].items():
                for i in range(count):
                    edges[edgeCount].append(sid)
            edgeCount += 1

        row, col = [], []
        for key, item in edges.items():
            col += [key] * len(item)
            row += item

        self.biadjacency = sparse.csr_matrix((np.ones(len(row), dtype=int), (row, col)))


    def constructBipartite(self):
        edgeCount = 0
        edges = {}

        for R2 in self.FL:
            edges[edgeCount] = sorted(self.FL[R2])
            edgeCount += 1
        #print(edges)
        row, col = [], []
        for key, item in edges.items():
            col += [key] * len(item)
            row += item
        #print(len(row), len(col))
        self.biadjacency = sparse.csr_matrix((np.ones(len(row), dtype=int), (row, col)))

    def clustering_KMeans(self, clusterNo):
        km = KMeans(clusterNo, precompute_distances=True)
        km.fit(self.normalized)

        self.labels_ = km.labels_[:len(self.classes)]

    def clustering(self, clusterNo):
        km = KMedoids(clusterNo, metric='precomputed', init='k-medoids++')
        km.fit(self.metric)

        self.labels_ = km.labels_[:len(self.classes)]

    def getDistance(self):
        self.metric = pairwise_distances(self.normalized)

    def getNormalizedLaplacianWithReturn(self, ipt, dim):
        self.constructBipartiteWithWeight()

        #use bipartite adjacency but apply normalization, and apply SVD on it.
        #to show that we can take advantages by setting bipartite
        spectral = BiSpectral(dim).fit(ipt)
        #we have two options: co-clustering or just normal clustering...
        normalized = spectral.row_embedding_

        #calculate transform matrix
        index = np.argsort(-spectral.solver.singular_values_)
        singular_values_ = spectral.solver.singular_values_[index[1:]]
        col_embedding_ = spectral.solver.right_singular_vectors_[:, index[1:]]
        #inverse the value
        singular = np.diag(1/singular_values_)

        transformation = np.dot(col_embedding_, singular)

        return normalized, transformation, singular_values_

    def calculateTransformMatrix(self, ipt, transformation, singular_values_, dim):
        adj = BiSpectral(dim).returnNormalized(ipt)
        normalized = np.dot(adj.sparse_mat.A, transformation)
        return normalized

    def getNormalizedLaplacian(self, dim):
        self.constructBipartiteWithWeight()

        #use bipartite adjacency but apply normalization, and apply SVD on it.
        #to show that we can take advantages by setting bipartite
        spectral = BiSpectral(dim).fit(self.biadjacency)
        #we have two options: co-clustering or just normal clustering...
        self.normalized = spectral.row_embedding_

        #L2 normalization ...
        norm = np.linalg.norm(self.normalized, ord=1, axis=1)
        norm[norm == 0.] = 1
        self.normalized /= norm[:, np.newaxis]
        self.normalized

    def calculatePurity(self):
        rst ={}
        for idx, val in enumerate(self.labels_):
            if val not in rst:
                rst[val] = []
            rst[val].append(self.classes[idx])
        sumVal = 0
        for i in rst.values():
            sumVal += max(Counter(i).values())
        return rst, sumVal/len(self.classes)

    def trial_KMeans(self, clusterNo, maxCount = 1):
        self.clustering_KMeans(clusterNo)
        rst, purity = self.calculatePurity()
        purities = [purity]
        count = 1

        while count < maxCount:
            max_purity = purity
            max_rst = rst
            #row_labels = self.clustering_inner(spectral, bipartite, clusterNo)
            self.clustering_KMeans(clusterNo)
            rst, purity = self.calculatePurity()
            count += 1
            purities.append(purity)
            if max_purity > purity:
                purity = max_purity
                rst = max_rst
        return rst, purities, max(purities), sum(purities)/maxCount

    def trial(self, clusterNo, maxCount = 1):
        self.clustering(clusterNo)
        rst, purity = self.calculatePurity()
        purities = [purity]
        count = 1

        while count < maxCount:
            max_purity = purity
            max_rst = rst
            #row_labels = self.clustering_inner(spectral, bipartite, clusterNo)
            self.clustering(clusterNo)
            rst, purity = self.calculatePurity()
            count += 1
            purities.append(purity)
            if max_purity > purity:
                purity = max_purity
                rst = max_rst

        return rst, purities, max(purities), sum(purities)/maxCount

    def ZEmbedding(self, printing=False):
        if printing == True:
            print("########## Z-EMBEDDING ##########")
            print("1-1. MINIMUM SUPPORT:", self.constraints["minSup"])
            print("1-2. MAXIMUM SUPPORT:", self.constraints["maxSup"])
            print("1-3. EPSILON CONSTRAINT:", self.constraints["epsilon"])
            print("1-4. GAP CONSTRAINT:", self.constraints["gap"])
            print("2. NUMBER OF E-SEQUENCES:", len(self.database.sequences))

        t1 = time.perf_counter()
        self.createGHashTable()
        t2 = time.perf_counter()

        # print("4. DFS time: ", t4 - t2)
        if printing==True:
            print("3. TOTAL COMPARISON COUNTS:", self.comparisoncount)
            print("4. TOTAL FREQUENT ARRANGEMENTS:", self.totalfrequency)
            print("5. TOTAL TIME CONSUMED:", t2 - t1)

        return self.comparisoncount, self.totalfrequency, t2 - t1, False, self.FL

def preprocess_class(filename):
    new_list = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        your_list = list(reader)

    for i in your_list:
        new_list.append(int(i[0]))

    return new_list

def NFoldClassification(algorithm, n=10, k=1, cl='rf', kernel='rbf', printing=False):
    t1 = time.perf_counter()
    kf = StratifiedKFold(n_splits=n, shuffle=True)
    kf.get_n_splits(algorithm.biadjacency)
    classes = np.array(algorithm.classes)
    count = 1
    scores=[]
    dim = algorithm.dim
    for train_index, test_index in kf.split(algorithm.biadjacency, classes):
        #print("TRIAL %d:" % count)
        count += 1
        X_train, X_test = algorithm.biadjacency[train_index], algorithm.biadjacency[test_index]
        y_train, y_test = classes[train_index], classes[test_index]
        #print("MAKING ADJACENCY MATRIX")
        train, transformation, singular = algorithm.getNormalizedLaplacianWithReturn(X_train, dim)

        #L2 Normalization
        norm = np.linalg.norm(train, axis=1)
        norm[norm == 0.] = 1
        train /= norm[:, np.newaxis]

        #print("FITTING")
        if cl == False:
            neigh = KNeighborsClassifier(n_neighbors=k)
        elif cl == 'rf' or cl == "RF":
            neigh = RandomForestClassifier()
        elif cl == 'svm' or cl == "SVM":
            neigh = svm.SVC(kernel=kernel)
        neigh.fit(train, y_train)
        #print("TESTING")

        #L2 Normalization
        test = algorithm.calculateTransformMatrix(X_test, transformation, singular, dim)
        norm = np.linalg.norm(test, axis=1)
        norm[norm == 0.] = 1
        test /= norm[:, np.newaxis]
        score = neigh.score(test, y_test)
        scores.append(score)
        if printing==True:
            print(score, end = ', ')
    if printing==True:
        print("AVG: ", sum(scores)/len(scores))
    t2 = time.perf_counter()
    if printing==True:
        print("FOLD AVG TIME:", (t2-t1)/n)
    return sum(scores)/len(scores), (t2-t1)/n

def sortTuple(tup):
    lst = len(tup)
    for i in range(0, lst):
        for j in range(0, lst-i-1):
            if (tup[j][-1] < tup[j + 1][-1]):
                temp = tup[j]
                tup[j]= tup[j + 1]
                tup[j + 1]= temp
    return tup

def gridSearch(filename, k=1, n=10, dim=8, cl=False, printing=False):

    tseq, tdis, tintv, aintv, avgtime, dataset = preprocess(filename)
    classname = filename.split("/")[0]+"/"+filename.split("/")[-1].split(".")[0]+"_CLASSES."+filename.split(".")[-1]
    classes = preprocess_class(classname)
    database = Database(dataset)

    #from 0.0 to 1.0
    gridMinSup = np.arange(0, 1.1, 0.1)
    gridMaxSup = np.arange(0, 1.1, 0.1)
    gridGap = np.arange(0, 1.1, 0.1)

    results = []
    #start gridSearch
    for minSup in gridMinSup:
        print(minSup*100,"% completed ...")
        for maxSup in gridMaxSup:
            #maxSup should be bigger than minSup
            if maxSup > minSup:
                for gap in gridGap:

                    #gap is proportion of the average time
                    constraints = makeConstraints([minSup, maxSup, 0, gap*avgtime], dataset)
                    algorithm = ZEmbedding(database, constraints, classes)
                    count, freq, timedelta, timeout, FL = algorithm.ZEmbedding(printing=printing)
                    try:
                        algorithm.getNormalizedLaplacian(dim)
                        rst = NFoldClassification(algorithm, k=k, n=n, dim=dim, cl=rf, printing=printing)
                        results.append((minSup, maxSup, gap, rst))
                    #error when there is no available instances
                    except:
                        continue

    results = sortTuple(results)
    print("TOP 10 GRID SEARCH RESULT")
    print(results[:10])
    return results

def createZEmbedding(filename, dim=8, minSup=0, maxSup=1, gap=1.0, printing=False):
    print("STEP 1: LOADING THE DATA")
    print("=========================")
    tseq, tdis, tintv, aintv, avgtime, dataset = preprocess(filename)
    classname = filename.split("/")[0]+"/"+filename.split("/")[-1].split(".")[0]+"_CLASSES."+filename.split(".")[-1]
    classes = preprocess_class(classname)
    print("TOTAL SEQUENCE:", tseq)
    print("TOTAL DISTINCT EVENTS:", tdis)
    print("TOTAL INTERVALS:", tintv)
    print("AVERAGE INTERVAL PER SEQ:", aintv)
    print("AVERAGE TIMESPAN:", avgtime)
    print("TEST WITH", filename, "DATASET")
    database = Database(dataset)
    print("=========================")
    print("STEP 2: MAKE A GRAPH AND APPLY SPECTRAL EMBEDDING")

    t1 = time.perf_counter()
    constraints = makeConstraints([minSup, maxSup, gap*avgtime], dataset)
    algorithm = ZEmbedding(database, constraints, classes)
    count, freq, timedelta, timeout, FL = algorithm.ZEmbedding(printing=printing)
    algorithm.getNormalizedLaplacian(dim)
    t2 = time.perf_counter()
    print("TOTAL TIME EMBEDDING: ", t2-t1)
    algorithm.dim = dim
    return algorithm

def NTrialClustering(algorithm, k, n=100):
    t1 = time.perf_counter()
    algorithm.getDistance()
    print("=========================")
    print("STEP 3: CALCULATE K-MEDOIDS")
    print("# OF GROUNT TRUTH CLUSTERS: ", k)
    print("# OF TRIALS: ", n)
    rst, purities, max_purity, mean_purity  = algorithm.trial(k, n)
    t2 = time.perf_counter()
    print("MAX PURITY: ", max_purity, "AVG PURITY: ", mean_purity)
    print("TOTAL AVG TIME KMEDOIDS: ", (t2-t1)/n)
    print("========K-Means========")
    t1 = time.perf_counter()
    rst, purities, max_purity, mean_purity  = algorithm.trial_KMeans(k, n)
    t2 = time.perf_counter()
    print("MAX PURITY: ", max_purity, "AVG PURITY: ", mean_purity)
    print("TOTAL AVG TIME KMEANS: ", (t2-t1)/n)
    print("=========================")
    return mean_purity, (t2-t1)/n
