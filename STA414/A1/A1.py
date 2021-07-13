# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:29:17 2021

@author: Harold
"""

import numpy as np
import matplotlib.pyplot as plt
import random

data_train = {'X': np.genfromtxt('data_train_X.csv', delimiter=','), 
              't': np.genfromtxt('data_train_Y.csv', delimiter=',')}
data_test = {'X': np.genfromtxt('data_test_X.csv', delimiter=','), 
              't': np.genfromtxt('data_test_Y.csv', delimiter=',')}



def shuffle_data(data):
    """
    Parameters
    ----------
    data : dictionary (bold_t, phi)
        Keys are 'X' and 't'. Contains ndarray of input vectors and 
        target vector.

    Returns
    -------
    data_shf
    returns its randomly permuted version along the samples. 
        preserves the same target-feature pairs.
    """
    # Shuffle the indices of X array.
    index = list(range(len(data['X'])))
    random.shuffle(index)
    # Create a new empty list of X and t 
    new_X = []
    new_t = []
    
    # rearrange the order of X and t array according to shuffled index
    for i in range(len(data['X'])):
        new_X.append(data['X'][index[i]])
        new_t.append(data['t'][index[i]])
        
    # convert the lists back into array and put them in new dictionary
    data_shf = {'X': np.array(new_X), 't': np.array(new_t)}
    return(data_shf)
    
def split_data(data, num_folds, fold):
    """
    Parameters
    ----------
    data : dictionary
        Keys are 'X' and 't'. Contains ndarray of input vectors and 
        target vector.
    num_folds : int
        number of partitions
    fold : int
        selected partition

    Returns
    -------
    data_fold: selected partition of data 
    data_rest: rest of partition of data
    """
    # fold_length is always an integer since we assume num_folds divides len(data)
    fold_length = len(data['X']) // num_folds
    
    # indices of the fold block
    fold_index = list(range((fold-1)*fold_length, fold*fold_length))
    
    # indices of all X array in data
    all_index = list(range(len(data['X'])))
    
    # Use list comprehension to remove fold index from all index
    rest_index = [item for item in all_index if item not in fold_index]
    
    
    data_fold = {'X': data['X'][(fold-1)*fold_length:fold*fold_length], 
                 't': data['t'][(fold-1)*fold_length:fold*fold_length]}
    
    rest_X = []
    rest_t = []
    for i in range(len(rest_index)):
        rest_X.append(data['X'][rest_index[i]])
        rest_t.append(data['t'][rest_index[i]])
    
    data_rest = {'X': np.array(rest_X), 't':np.array(rest_t)}
    #rest_X = data['X'][0:(folds - 1)*fold_length] + data['X'][folds*fold_length:]
    return(data_fold, data_rest)


def train_model(data, lambd):
    """
    Parameters
    ----------
    data : dictionary
        Keys are 'X' and 't'. Contains ndarray of input vectors and 
        target vector.
    lambd : float
        penalty level coefficient.

    Returns
    -------
    model: coefficients of ridge regression.
    """
    # Store number of observations to obs
    obs = len(data['X'])
    # Store number of parameters to var
    var = len(data['X'][0])
    
    phi = np.reshape(data['X'], (obs, var))
    # Reshape X data array to matrix of obs by var
    # Stack target array to column vector
    t = np.vstack(data['t'])
    
    phiTphi = np.dot(np.transpose(phi), phi)
    
    # Compute w_hat
    model = np.dot(np.linalg.inv(phiTphi + lambd*np.identity(var)), np.dot(np.transpose(phi), t))
    
    # Reshape w_hat back into row vector
    model = np.hstack(model)
    return(model)

def predict(data, model):
    """
    Parameters
    ----------
    data : dictionary
        Keys are 'X' and 't'. Contains ndarray of input vectors and 
        target vector.
    model : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    # Store number of observations to obs
    obs = len(data['X'])
    # Store number of parameters to var
    var = len(data['X'][0])
    
    phi = np.reshape(data['X'], (obs, var))
    
    predictions = np.dot(phi, model)
    return(predictions)



def loss(data, model):
    """
    

    Parameters
    ----------
    data : dictionary
        Keys are 'X' and 't'. Contains ndarray of input vectors and 
        target vector.
    model : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    # Observed target vector
    t = data['t']
    
    # Compute predicted target values
    phi_w = predict(data, model)
    
    return(sum(np.square(t-phi_w))/len(data['X']))    


def cross_validation(data, num_folds, lambd_seq):
    """
    Parameters
    ----------
    data : dictionary
        Keys are 'X' and 't'. Contains ndarray of input vectors and 
        target vector.
    num_folds : int
        number of CV folds.
    lambd_seq : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    cv_error = []
    
    data = shuffle_data(data)
    for lambd in lambd_seq:
        cv_loss_lmd = 0
        for fold in range(1,num_folds+1):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error.append(cv_loss_lmd / num_folds)
    return(cv_error)






# Set random seed for plotting
np.random.seed(1)
# Make 50 intervals from 0.02 to 1.5
lambdas = np.linspace(0.02, 1.5, num=50)

# Compute 5 fold and 10 fold error rate
cross_5_error = cross_validation(data_train, 5, lambdas)
cross_10_error = cross_validation(data_train, 10, lambdas)

train_error = []
test_error = []
for lambd in lambdas:
    # Compute 50 model parameters from 50 lambdas
    model = train_model(data_train, lambd)
    # Compute training error using loss function
    train_error.append(loss(data_train, model))
    # Compute testing error using loss function
    test_error.append(loss(data_test, model))
    
# Finding the lambda proposed by your cross validation procedure.
print(lambdas[test_error.index(min(test_error))])

plt.plot(lambdas, cross_5_error, 'b', label = "5 fold Error")
plt.plot(lambdas, cross_10_error, 'g', label = "10 fold Error")
plt.plot(lambdas, train_error, 'k', label = "Training Error")
plt.plot(lambdas, test_error, 'r', label="Testing Error")
plt.xlabel("Lambdas")
plt.ylabel("Error")
plt.legend(loc='upper right')


