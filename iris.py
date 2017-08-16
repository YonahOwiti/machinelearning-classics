'''
Created on Aug 16, 2017

@author: Varela

'''
import numpy as np 
import matplotlib.pyplot as plt 

from utils import get_iris  

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
	# detect ='Iris-versicolor' 
	# detect ='Iris-setosa'
	detect ='Iris-virginica'
	X, T  = get_iris(detect=detect)
	X, T  = np.shuffle(X,T)

	N, D  = X.shape 
	X 		= np.concatenate((np.ones((N,1)), X), axis=1, ) 
	T = T.astype(np.int32)
	X = X.astype(np.float32)
	D+=1

	# params
	lr = 5e-4
	max_iteration=1000
	W  		= np.random.randn(D) / np.sqrt(D)
	cost 	= []
	error = [] 
	for i in xrange(max_iteration):
		Y = forward(W, X)
		cost.append(cross_entropy(T,Y))
		error.append(error_rate(T,Y))

		W += lr*X.T.dot(T-Y)

		if i % 5 == 0:
			print "i=%d\tcost=%.3f\terror=%.3f" % (i,cost[-1],error[-1])

	if i % 5 == 0:
			print "i=%d\tcost=%.3f\terror=%.3f" % (i,cost[-1],error[-1])
					
	print "Final weight:", W 
	print T 
	print np.round(Y)

	

	plt.title('logistic regression ' + detect)
	plt.xlabel('iterations')
	plt.ylabel('cross entropy')
	plt.plot(cost)
	plt.show()

	plt.title('logistic regression ' + detect)
	plt.xlabel('iterations')
	plt.ylabel('error rate')
	plt.plot(error)
	plt.show()


if __name__ == '__main__':
	main()
	