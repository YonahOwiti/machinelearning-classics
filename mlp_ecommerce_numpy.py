'''
Created on Aug 20, 2017

@author: Varela

motivation: Multilayer perceptron for ecommerce data
status:	not converging			
'''
import numpy as np 
import matplotlib.pyplot as plt 

from utils import get_ecommerce, sigmoid 



def forward(W1, b1, W2, b2, X):
	Z = sigmoid(X.dot(W1) + b1)
	A = Z.dot(W2) + b2
	expA = 	np.exp(A) 
	Y = expA / expA.sum(axis=1, keepdims=True)	
	return Y, Z 


# determine the classification rate
# num correct / num total
def classification_rate(Y, P):
	n_correct = 0
	n_total = 0
	for i in xrange(len(Y)):
		n_total += 1
		if Y[i] == P[i]:
			n_correct += 1
	return float(n_correct) / n_total

def derivative_w2(Z, T, Y):
	return Z.T.dot(T-Y)

def derivative_b2(T,Y):
	return (T-Y).sum(axis=0)
	# return (T-Y).sum(axis=1)

def derivative_w1(X, Z, T, Y, W2):
	dZ = (T-Y).dot(W2.T)*Z*(1-Z)
	return X.T.dot(dZ)

def derivative_b1(T, Y, W2, Z):
	return ((T-Y).dot(W2.T) * Z * (1-Z)).sum(axis=0)

def cost(T, Y):	
	tot = T * np.log(Y)
	return tot.sum()

def main(): 
	X, Y  = get_ecommerce(user_action=None)
	
	# Define dimensions 
	N, D  = X.shape 	

	M = 10
	K = 4 

	#Input weights - variables
	# one-hot-encodefy-Y
	T = np.zeros((N, K), dtype=np.int32)
	for i in xrange(N):
		T[i, Y[i]] = 1


	W1 = np.random.randn(D,M) / np.sqrt(M + D)
	b1 = np.random.randn(M) / np.sqrt(M)

	W2 = np.random.randn(M,K) / np.sqrt(M + K)
	b2 = np.random.randn(K)   / np.sqrt(K)

	# Running variables 
	lr = 5e-8
	max_iteration=15000	
	costs 	= np.zeros((max_iteration,), dtype=np.float32)
	errors  = np.zeros((max_iteration,), dtype=np.float32)

	for i in xrange(max_iteration):
		output, hidden = forward(W1, b1, W2, b2, X)

		P = np.argmax(output, axis=1)
		r = classification_rate(Y, P)
		c = cost(T, output)

		if i % 100 == 0:		
			print "i=%d\tcost=%.3f\terror=%.3f" % (i,c,r)

		W2 += lr * derivative_w2(hidden, T, output)		
		b2 += lr * derivative_b2(T, output)
		# derivative_w1(X, Z, T, Y, W2)
		W1 += lr * derivative_w1(X,hidden, T, output,W2)
		b1 += lr * derivative_b1(T, output, W2, hidden)

		costs[i] = c
		errors[i] = r
		# costs.append(c)
		# errors.append(r)

	plt.title('Ecommerce costs')	
	plt.plot(costs)	
	plt.show()
	plt.title('Ecommerce error rate')
	plt.plot(errors)	
	plt.show()




if __name__ == '__main__':
	main()	