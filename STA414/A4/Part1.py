# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 17:17:57 2021

@author: Harold
"""


#%matplotlib inline
import scipy
import numpy as np
import itertools
import matplotlib.pyplot as plt


# TODO: Run this cell to generate the data
num_samples = 400
cov = np.array([[1., .7], [.7, 1.]]) * 10
mean_1 = [.1, .1]
mean_2 = [6., .1]

x_class1 = np.random.multivariate_normal(mean_1, cov, num_samples // 2)
x_class2 = np.random.multivariate_normal(mean_2, cov, num_samples // 2)
xy_class1 = np.column_stack((x_class1, np.zeros(num_samples // 2)))
xy_class2 = np.column_stack((x_class2, np.ones(num_samples // 2)))
data_full = np.row_stack([xy_class1, xy_class2])
np.random.shuffle(data_full)
data = data_full[:, :2]
labels = data_full[:, 2]

plt.scatter(x_class1[:,0], x_class1[:,1], marker="x")
plt.scatter(x_class2[:,0], x_class2[:,1])




def cost(data, R, Mu):
    N, D = data.shape
    K = Mu.shape[1]
    J = 0
    for k in range(K):
        J += np.dot(np.linalg.norm(data - np.array([Mu[:, k], ] * N), axis=1)**2, R[:, k])
    return J

# TODO: K-Means Assignment Step
def km_assignment_step(data, Mu):
    """ Compute K-Means assignment step
    
    Args:
        data: a NxD matrix for the data points
        Mu: a DxK matrix for the cluster means locations
    
    Returns:
        R_new: a NxK matrix of responsibilities
    """
    
    # Fill this in:
    N, D = data.shape # Number of datapoints and dimension of datapoint
    K = Mu.shape[1] # number of clusters
    r = np.zeros((N, K))
    
    for k in range(K):
        # r[:, k] = ...
        r[:, k] = np.linalg.norm(data - np.array([Mu[:, k], ] * N), axis=1)**2
        
    # arg_min = ... # argmax/argmin along dimension 1
    # axis = 1 -> by rows
    arg_min = np.argmin(r, axis = 1)
    
    
    # R_new = ... # Set to zeros/ones with shape (N, K)
    # R_new[..., ...] = 1 # Assign to 1
    
    R_new = np.zeros((N,K))
    R_new[np.array(range(N)), arg_min] = 1
    
    return R_new

# TODO: K-means Refitting Step
def km_refitting_step(data, R, Mu):
    """ Compute K-Means refitting step.
    
    Args:
        data: a NxD matrix for the data points
        R: a NxK matrix of responsibilities
        Mu: a DxK matrix for the cluster means locations
    
    Returns:
        Mu_new: a DxK matrix for the new cluster means locations
    """
    # N, D = data.shape # Number of datapoints and dimension of datapoint
    # K = Mu.shape[1]  # number of clusters
    # Mu_new = ...
    
    # axis = 0 will fix the column
    Mu_new = np.dot(data.T, R)/np.sum(R, axis = 0)
    return Mu_new


N, D = data.shape
K = 2
max_iter = 100
class_init = np.random.binomial(1., .5, size=N)
R = np.vstack([class_init, 1 - class_init]).T

Mu = np.zeros([D, K])
Mu[:, 1] = 1.
R.T.dot(data), np.sum(R, axis=0)

cost_list = []
for it in range(max_iter):
    R = km_assignment_step(data, Mu)
    Mu = km_refitting_step(data, R, Mu)
    cost_list.append(cost(data, R, Mu))
    print(it, cost(data, R, Mu))

class_1 = np.where(R[:, 0])
class_2 = np.where(R[:, 1])

class_1_val = data[class_1]
class_2_val = data[class_2]


plt.scatter(class_1_val[:,0], class_1_val[:,1], marker="x")
plt.scatter(class_2_val[:,0], class_2_val[:,1])


plt.plot(range(max_iter), cost_list)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")




misclass_1 = sum(data_full[class_1][:,2])
misclass_2 = sum(data_full[class_2][:,2] == 0)

mis_error = (misclass_1 + misclass_2)/400



x = data_full[class_1][:,2]


data_full

