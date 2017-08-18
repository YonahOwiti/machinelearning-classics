'''
Created on Aug 17, 2017

@author: Varela

motive: Logistic regression for fer2013.csv dataset
'''

import numpy as np 
import matplotlib.pyplot as plt 

from utils import get_facialexpression, error_rate, cross_entropy,  forward   

def main():
	
	X, T  = get_facialexpression(balance_ones=True)
	# X, T  = np.shuffle(X,T)
	label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
	# klass =3  error_rate=0.0
	# klass =4  error_rate=0.0
	klass =5
	N, D  = X.shape 
	X 		= np.concatenate((np.ones((N,1)), X), axis=1, ) 
	T 		= T.astype(np.int32)
	X 		= X.astype(np.float32)
	#Fix for forecasting on one image
	T[T==klass]=1
	T[T!=klass]=0


	D+=1

	# params
	lr = 5e-7
	max_iteration=12000
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

	

	plt.title('logistic regression ' + label_map[klass])
	plt.xlabel('iterations')
	plt.ylabel('cross entropy')
	plt.plot(cost)
	plt.show()

	plt.title('logistic regression ' + label_map[klass])
	plt.xlabel('iterations')
	plt.ylabel('error rate')
	plt.plot(error)
	plt.show()


if __name__ == '__main__':
	main()
