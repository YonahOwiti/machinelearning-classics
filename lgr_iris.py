'''
Created on Aug 16, 2017

@author: Varela

motive: Logistic regression for iris dataset
'''
import numpy as np 
import matplotlib.pyplot as plt 

from utils import get_iris  
from utils_data import *

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


def main():
	
	X, T  = get_iris()
	T, label_map = class2numeric(T)
	
	detect = 1	
	T = class1detect(T,detect)
	

	print "detecting:", label_map[detect]

	N, D  = X.shape 
	X 		= np.concatenate((np.ones((N,1)), X), axis=1, ) 
	T 		= T.astype(np.int32)
	X 		= X.astype(np.float32)
	D+=1

	Xtrain, Ttrain, Xtest, Ttest = split2test(X, T, perc=0.1, balanced=True)

	# params
	lr = 5e-4
	max_iteration=1000
	W  		= np.random.randn(D) / np.sqrt(D)
	cost 	= []
	error = [] 
	for i in xrange(max_iteration):
		Ytrain = forward(W, Xtrain)
		cost.append(cross_entropy(Ttrain,Ytrain))
		error.append(error_rate(Ttrain,Ytrain))

		W += lr*Xtrain.T.dot(Ttrain-Ytrain)

		if i % 5 == 0:
			print "train: i=%d\tcost=%.3f\terror=%.3f" % (i,cost[-1],error[-1])
			Ytest = forward(W, Xtest)
			cost_test 	= cross_entropy(Ttest,Ytest)
			error_test  = error_rate(Ttest,Ytest)
			print "validation: i=%d\tcost=%.3f\terror=%.3f" % (i,cost_test,error_test)

	if i % 5 == 0:
			print "i=%d\tcost=%.3f\terror=%.3f" % (i,cost_test,error_test)
					
	print "Final weight:", W 
	print Ttest 
	print np.round(Ytest)

	

	plt.title('logistic regression ' + label_map[detect])
	plt.xlabel('iterations')
	plt.ylabel('cross entropy')
	plt.plot(cost)
	plt.show()

	plt.title('logistic regression ' + label_map[detect])
	plt.xlabel('iterations')
	plt.ylabel('error rate')
	plt.plot(error)
	plt.show()


if __name__ == '__main__':
	main()
	