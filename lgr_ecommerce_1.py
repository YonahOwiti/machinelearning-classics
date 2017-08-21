'''
Created on Aug 21, 2017

@author: Varela

motive: Logistic regression + softmax
course url : https://www.udemy.com/data-science-deep-learning-in-python/learn/v4/t/lecture/5284622?start=0				
'''
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle 
from utils import get_ecommerce, y2indicator, softmax, classification_rate 

def forward(W, b, X):
	return softmax(X.dot(W) + b)

def predict(P_Y_given_X):
	return np.argmax(P_Y_given_X, axis=1)

def cross_entropy(T, pY): 	
	return -np.mean(T*np.log(pY))



def main():
	
	X, Y  = get_ecommerce(user_action=None)
	X, Y  = shuffle(X,Y)


	N, D  = X.shape 	
	Y     = Y.astype(np.int32)
	K     = len(np.unique(Y))
	
	Ntrain = N -100+1 
	Xtrain, Ytrain = X[:Ntrain,: ], Y[:Ntrain]
	Ytrain_ind = y2indicator(Ytrain, K)

	Ntest  = 100
	Xtest, Ytest  = X[-Ntest:,: ], Y[-Ntest:]
	Ytest_ind = y2indicator(Ytest, K)	
	


	# params
	lr = 5e-3
	max_iteration=10000
	W  		= np.random.randn(D,K) / np.sqrt(D + K)
	b     = np.zeros(K)
	
	train_costs = [] 
	test_costs = []

	for i in xrange(max_iteration):
		pYtrain, pYtest = forward( W, b, Xtrain), forward( W, b, Xtest)
		# Ytrain  = predict(pYtrain)
		ctrain = cross_entropy(Ytrain_ind,pYtrain)
		ctest  = cross_entropy(Ytest_ind, pYtest)

		train_costs.append(ctrain)
		test_costs.append(ctest)


		W -= lr*Xtrain.T.dot(pYtrain-Ytrain_ind)
		b -= lr*(pYtrain-Ytrain_ind).sum(axis=0)

		if i % 1000 == 0:
			print "i=%d\ttrain cost=%.3f\ttest error=%.3f" % (i,ctrain,ctest)
			
	print "i=%d\ttrain cost=%.3f\ttest error=%.3f" % (max_iteration,ctrain,ctest)
	print "Final train classification rate", classification_rate(Ytrain, predict(pYtrain))
	print "Final test  classification rate", classification_rate(Ytest, predict(pYtest))

	plt.title('logistic regression + softmax')
	plt.xlabel('iterations')
	plt.ylabel('training costs')
	legend1, = plt.plot(train_costs, label='train cost')
	legend2, = plt.plot(test_costs, label='test cost')
	plt.legend([legend1, legend2,])
	plt.show()


if __name__ == '__main__':
	main()
	