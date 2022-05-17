# This file provides Sk-learn functionality for reducing the feature dimensionality of multi-dimsional time series data as a pre-processing step in calculating Tononi Phi.
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
import csv
from sklearn import decomposition
import phi_params_27Apr22 as conf

from sklearn.decomposition import FastICA, PCA, fastica
from numpy import genfromtxt

# The compute_ica function reads in a file name. The input file should be a csv or tsv array-like file of shape (n_samples, n_features).compute_ica outputs a pair [S_, num_of_nodes] in which S_ is the best fit ICA transform and num_of_nodes is the number of nodes leading to the smallest sum of sqaure residuals (ssr). The number of nodes is currently hard-coded as being in the range [3, max_nodes].
def compute_ica(file, starting_value, int_len, max_nodes):
    S = genfromtxt(file, delimiter = conf.delim)
    S = S[starting_value: starting_value + int_len -1]
    print(S)
#    S = genfromtxt(rows[starting_value:starting_value+int_len-1], delimiter = conf.delim)
#    S = S[starting_value: starting_value + int_len -1]
    ica = None
    ssr = np.zeros(max_nodes + 1)
    # Determine number of nodes that minimizes sum of the squares of the errors
    for loc in range (max_nodes + 1):
        dim = loc+3
#        ica = decomposition.FastICA(n_components = dim, max_iter = 1000, tol = 1e-02)
#        S_ = ica.fit_transform(S)
#        A_ = ica.mixing_.T
        #np.allclosei(X, np.dot(S_, A_) + ica.mean_)
        [K, W, S_] = fastica(S, n_components = dim, max_iter = 1000, tol = 1e-02)
        w = np.dot(W.T, K)
        #A = w.T*(w*w.T).I
        wwt = np.dot(w, w.T)
        #print(wwt.shape)
        A = np.dot(w.T, np.linalg.inv(wwt))
        X = np.matmul(A, S_.T)
#        print(loc)
        ssr[loc] = ((X-S.T) ** 2).sum()
        #        f.write("dimension = " + str(dim) + " and ssr = " + str(ssr[loc]) + "\n")
        #    mn = np.min(ssr)
    print(ssr)
    num_of_nodes = np.argmin(ssr) + 3
    #f.write("minimum is " + str(mn) + " at dim = " + str(ind+3))
    #f.close()
    ica = decomposition.FastICA(n_components = num_of_nodes, max_iter = 1000, tol = 1e-02)
    S_ = ica.fit_transform(S)
    return([S_, num_of_nodes])

# filename is the name of the file passed to the compute_ica function and set max_nodes
##filename = "/Users/moikle_admin/Research/SingularityNET/Phi+Reputation/equilDefault9conserv1/rankHistory_r_20_0.1_test.csv"
#max_nodes = 20

# Here we call the compute_ica function with arguments filename and max_nodes
#[S_, num_of_nodes] = compute_ica(filename, max_nodes)
#with open('/Users/moikle_admin/Research/SingularityNET/Phi+Reputation/equilDefault9conserv1/S_.csv', 'w') as csvfile:
#    writer = csv.writer(csvfile)
#    writer.writerows(S_)

# In the following codes section, we write all the phi parameters, including the num_of_nodes that minimizes ssr, into the file phi_params.py
# This is a definite hack. The ICA code will first provide a dimensionality reduction step for the number of relevant features to be used in the time series. This is needed since Queyranne's algorithm grows as the cube of the number of features. There is a balance required between the number of features used in the Tononi Phi calculations, and the accuracy of the result, especially since we are ultimately seeking real-time Phi outputs for demonstration purposes.
# What is still unclear, is whether the number of relevant nodes should be determined first for an entire time-series, or whether it should be determined "on-the-fly" adjusting to the most relevant features over most recent time period of length int_len.
# phi_params.py will be read in by the main Tononi Phi file and num_of_nodes will be used to deteremine the MIP.

