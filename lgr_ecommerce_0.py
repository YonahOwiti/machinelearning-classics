'''
Created on Aug 16, 2017

@author: Varela

motive: Logistic regression for ecommerce data
				1 class forecast only 
'''
import numpy as np 
import matplotlib.pyplot as plt 

from utils import get_ecommerce, forward, cross_entropy, error_rate 

def main():
	user_action=3
	X, T  = get_ecommerce(user_action=user_action)
	# X, T  = np.shuffle(X,T)

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

	plt.title('logistic regression user_action=%d' % (user_action))
	plt.xlabel('iterations')
	plt.ylabel('cross entropy')
	plt.plot(cost)
	plt.show()

	plt.title('logistic regression user_action=%d' % (user_action))
	plt.xlabel('iterations')
	plt.ylabel('error rate')
	plt.plot(error)
	plt.show()


if __name__ == '__main__':
	main()
	