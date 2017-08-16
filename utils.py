'''
Created on Aug 16, 2017

@author: Varela

'''

import numpy as np 
import pandas as pd 

def error_rate(T,Y):
	return np.mean(np.round(Y)!=T)

# calculate the cross-entropy error
def cross_entropy(T, Y):
    E = 0
    for i in xrange(T.shape[0]):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E
	
def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))

def forward(W, X):
	return sigmoid(X.dot(W))

def get_iris(detect='Iris-setosa'):
	X = []; Y =[]
	if not( detect in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']): 
		raise ValueError( 'value to detect - %s invalid' % (detect))
	df = pd.read_csv('./datasets/iris/iris.data', sep=',')
	X=  df.values[:,:-1]
	Y=  df.values[:,-1] == detect

	return X, Y 

def get_ecommerce(user_action=1):
	X = []; Y =[]
	if not( user_action in [1,2,3]): 
		raise ValueError( 'value to user_action - %d invalid' % (user_action))

	df = pd.read_csv('./datasets/ecommerce/ecommerce_data.csv', sep=',', header=0)
	X=  df.values[:,:-1]
	Y=  np.array(df.values[:,-1] == user_action, dtype=np.int32)
	
	return X, Y 
