'''
Created on Aug 20, 2017

@author: Varela

motivation: Multilayer perceptron for ecommerce data
status:	not converging			
'''
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle 	
from utils import get_ecommerce, softmax, y2indicator, error_rate, classification_rate 

def predict(P_Y_given_X):
	return np.argmax(P_Y_given_X, axis=1)


def forward(W1, b1, W2, b2, X):
	Z = np.tanh(X.dot(W1) + b1)
	# A = Z.dot(W2) + b2
	# expA = 	np.exp(A) 
	# Y = expA / expA.sum(axis=1, keepdims=True)	
	return softmax(Z.dot(W2) + b2), Z 


# determine the classification rate
# num correct / num total
# def classification_rate(Y, P):
# 	n_correct = 0
# 	n_total = 0
# 	for i in xrange(len(Y)):
# 		n_total += 1
# 		if Y[i] == P[i]:
# 			n_correct += 1
# 	return float(n_correct) / n_total

# def derivative_w2(Z, T, Y):
# 	return Z.T.dot(T-Y)

# def derivative_b2(T,Y):
# 	return (T-Y).sum(axis=0)	

# def derivative_w1(X, Z, T, Y, W2):
# 	dZ = (T-Y).dot(W2.T)*Z*(1-Z)
# 	return X.T.dot(dZ)

# def derivative_b1(T, Y, W2, Z):
# 	return ((T-Y).dot(W2.T) * Z * (1-Z)).sum(axis=0)

# def cost(T, Y):	
# 	tot = T * np.log(Y)
# 	return tot.sum()

def cross_entropy(T, pY):
	return  -np.sum(T * np.log(pY))


def main(): 
	X, Y  = get_ecommerce(user_action=None)
	X, Y  = shuffle(X, Y)

	# Define dimensions 
	N, D  = X.shape 	
	M = 5
	K = len(np.unique(Y))

	Ntrain = N -100+1 
	Xtrain, Ytrain = X[:Ntrain,: ], Y[:Ntrain]
	Ytrain_ind = y2indicator(Ytrain, K)

	Ntest  = 100
	Xtest, Ytest  = X[-Ntest:,: ], Y[-Ntest:]
	Ytest_ind = y2indicator(Ytest, K)	
	

	W1 = np.random.randn(D,M) / np.sqrt(M + D)
	b1 = np.random.randn(M) / np.sqrt(M)

	W2 = np.random.randn(M,K) / np.sqrt(M + K)
	b2 = np.random.randn(K)   / np.sqrt(K)

	# Running variables 
	lr = 5e-4
	max_iteration=100000	
	
	train_costs = []
	test_costs = []
	train_errors = []
	test_errors = []
	for i in xrange(max_iteration):
		pYtrain, Ztrain = forward(W1, b1, W2, b2, Xtrain)
		pYtest, Ztest   = forward(W1, b1, W2, b2, Xtest)

		ctrain = cross_entropy(Ytrain_ind,pYtrain)
		ctest  = cross_entropy(Ytest_ind, pYtest)

		etrain = error_rate(predict(pYtrain), Ytrain)
		etest  = error_rate(predict(pYtest), Ytest)

		train_costs.append(ctrain)
		test_costs.append(ctest)
		train_errors.append(etrain)
		test_errors.append(etest)

		

		if i % 1000 == 0:
			print "i=%d\ttrain cost=%d\ttest cost=%d\ttrain error=%0.3f" % (i,int(ctrain),int(ctest),etrain)

		W2 -= lr * Ztrain.T.dot(pYtrain - Ytrain_ind)
		b2 -= lr * (pYtrain - Ytrain_ind).sum(axis=0)
		# derivative_w1(X, Z, T, Y, W2)
		W1 -= lr * Xtrain.T.dot((pYtrain-Ytrain_ind).dot(W2.T)*Ztrain*(1-Ztrain))
		b1 -= lr * ((pYtrain-Ytrain_ind).dot(W2.T)*Ztrain*(1-Ztrain)).sum(axis=0)

		

	print "i=%d\ttrain cost=%.3f\ttest error=%.3f" % (max_iteration,ctrain,ctest)
	print "Final train classification rate", classification_rate(Ytrain, predict(pYtrain))
	print "Final test  classification rate", classification_rate(Ytest, predict(pYtest))
		
	plt.title('Multi layer perceptron: Costs')
	plt.xlabel('iterations')
	plt.ylabel('costs')
	legend1, = plt.plot(train_costs, label='train cost')
	legend2, = plt.plot(test_costs, label='test cost')
	plt.legend([legend1, legend2,])
	plt.show()


	plt.title('Multi layer perceptron: Error rates')
	plt.xlabel('iterations')
	plt.ylabel('erro rates')
	legend1, = plt.plot(train_costs, label='train error')
	legend2, = plt.plot(test_costs, label='test error')
	plt.legend([legend1, legend2,])
	plt.show()
	plt.xlabel('iterations')
	plt.ylabel('training costs')
	legend1, = plt.plot(train_costs, label='train cost')
	legend2, = plt.plot(test_costs, label='test cost')
	plt.legend([legend1, legend2,])
	plt.show()




if __name__ == '__main__':
	main()	