'''
Created on Aug 20, 2017

@author: Varela

motivation: Multilayer perceptron for ecommerce data + theano
	status:	not converging			
'''
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle 	
from utils import get_ecommerce, softmax, y2indicator, error_rate, classification_rate, sigmoid  

import theano
import theano.tensor as T

def main(): 
	X, Y  = get_ecommerce(user_action=None)
	X, Y  = shuffle(X, Y)

	# Running variables 
	learning_rate = 5e-4
	max_iterations=10000	

	# Define dimensions 
	N, D  = X.shape 	
	M = 5
	K = len(np.unique(Y))

	Ntrain = N -100
	Xtrain, Ytrain = X[:Ntrain,: ], Y[:Ntrain]
	Ytrain_ind = y2indicator(Ytrain, K)
	
	Ntest  = 100
	Xtest, Ytest  = X[-Ntest:,: ], Y[-Ntest:]
	Ytest_ind = y2indicator(Ytest, K)	
	

	W1_init = np.random.randn(D,M) / np.sqrt(M + D)
	b1_init = np.random.randn(M) / np.sqrt(M)

	W2_init = np.random.randn(M,K) / np.sqrt(M + K)
	b2_init = np.random.randn(K)   / np.sqrt(K)

	#Define theano shared 
	W1  = theano.shared(W1_init,'W1')
	b1  = theano.shared(b1_init,'b1')
	W2  = theano.shared(W2_init,'W2')
	b2  = theano.shared(b2_init,'b2')


	#Define constant tensor matrices
	thX = T.matrix('X')
	thT = T.matrix('T')

	#Define cost 
	thZ = sigmoid(thX.dot(W1) + b1)
	thY = softmax(thZ.dot(W2) + b2)

	cost  = -(thT * np.log(thY) +(1-thT) * np.log(1- thY)  ).sum()
	prediction  = T.argmax(thY,axis=1) 

	#Define updates
	W1_update = W1 - learning_rate*T.grad(cost, W1)
	b1_update = b1 - learning_rate*T.grad(cost, b1)
	W2_update = W2 - learning_rate*T.grad(cost, W2)
	b2_update = b2 - learning_rate*T.grad(cost, b2)


	train = theano.function(
		inputs=[thX, thT],
		updates = [(W1, W1_update), (b1, b1_update),(W2, W2_update), (b2, b2_update)],
	)
	predict = theano.function(
		inputs=[thX, thT],
		outputs=[cost, prediction],
	)

	LL = [] 
	train_errors = [] 
	test_errors = [] 
	train_costs = [] 
	test_costs = [] 
	for i in xrange(max_iterations):
		train(Xtrain, Ytrain_ind)		
		if i % 10 == 0:
			c, pYtrain = predict(Xtrain, Ytrain_ind)
			err = error_rate(Ytrain, pYtrain)
			train_costs.append(c)
			train_errors.append(err)

			c, pYtest = predict(Xtest, Ytest_ind)
			err = error_rate(Ytest, pYtest)
			test_costs.append(c)
			test_errors.append(err)
			print "i=%d\tc=%.3f\terr==%.3f\t" % (i,c,err)


	print "i=%d\tc=%.3f\terr==%.3f\t" % (max_iterations,c,err)

	
	print "Final train classification rate", classification_rate(Ytrain, pYtrain)
	print "Final test  classification rate", classification_rate(Ytest,  pYtest)
		
	plt.title('Multi layer perceptron: Costs')
	plt.xlabel('iterations')
	plt.ylabel('costs')
	legend1, = plt.plot(train_costs, label='train cost')
	legend2, = plt.plot(test_costs, label='test cost')
	plt.legend([legend1, legend2,])
	plt.show()


	plt.title('Multi layer perceptron: Error rates')
	plt.xlabel('iterations')
	plt.ylabel('error rates')
	legend1, = plt.plot(train_errors, label='train error')
	legend2, = plt.plot(test_errors, label='test error')
	plt.legend([legend1, legend2,])
	plt.show()
	


if __name__ == '__main__':
	main()	