'''
Created on Aug 19, 2017

@author: Varela

motive: Applies K class classification / lazyprogrammer's implementation
course url: https://www.udemy.com/data-science-logistic-regression-in-python/learn/v4/t/lecture/5217332?start=0

'''


import numpy as np 
import matplotlib.pyplot as plt 
import sys 

from sklearn.utils import shuffle 
from utils import sigmoid, error_rate, get_facialexpression, cross_entropy 
from utils_data import class1detect

class LogisticModelK(object):
	'''
		Multiclass perceptron
	'''
	def __init__(self, label=[]):
		self.label= label 
		pass 

	def fit(self, X, Y, learning_rate=1e-6, reg=0, epochs=12000, show_figure=False):
		X, Y  = shuffle(X, Y)

		Xvalid, Yvalid = X[-1000:,:], Y[-1000:]
		X, Y  = X[:-1000,:], Y[:-1000]
		

		K = len(set(Y))		
		N, D  = X.shape
		
		Yind_valid = np.zeros((1000,K),dtype=np.int32)
		Yind = np.zeros((N,K),dtype=np.int32)
		Yind_valid[np.arange(1000),Yvalid]=1
		Yind[np.arange(N),Y]=1

		self.W = np.random.randn(D, K) / np.sqrt(D + K)
		self.b = 0 

		costs = [] 
		best_validation_error = 1
		for i in xrange(epochs):
			for j in xrange(N):
				xj = X[j,:].T 
				yj = Y[j]
				
				yp = np.argmax((self.W.T).dot(xj),axis=0)
				
				# gradient descent step				
				self.W[:,yj] += (xj + reg*self.W[:, yj])
				self.W[:,yp] -= (xj + reg*self.W[:, yp])
				# self.b -= learning_rate *((pY-Y).sum() 	+ reg*self.b)

				if i % 20 == 0:
					import code; code.interact(local=dict(globals(), **locals()))
					pYvalid = self.forward(Xvalid)
					# c = sigmoid_cost(Yvalid, pYvalid)
					c = cross_entropy(Yind_valid, pYvalid) 
					costs.append(c)	
					e = error_rate(Yvalid, pYvalid)
					sys.stdout.write( "i:%s\tcost:%.4f\terror:%.4f\t\r" % (format(i,'04d'),c,e))
					sys.stdout.flush()
					# print "i", i, "cost:", c, "error", e
					if e < best_validation_error:
						best_validation_error = e
		print "best_validation_error:", best_validation_error

		if show_figure:
			plt.plot(costs)
			plt.show()
	
	def forward(self, X):			
		return X.dot(self.W)
  
	def predict(self, X):
		pY = self.forward(X)		
		return np.argmax(pY, axis=0)

	def score(self, X, Y):
		prediction = self.predict(X)
		return 1 - error_rate(Y, prediction)

def main():
	print 'Loading ...'
	X, Y = get_facialexpression(balance_ones=True)

 
 	label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
	print Y.shape 
	print Y

	# detect=1
	# print 'detecting class', detect
	# Y = class1detect(Y, detect)
	
	model = LogisticModelK(label=label_map)
	model.fit(X, Y, epochs=200, show_figure=True)
	model.score(X, Y)



if __name__ == '__main__':
	main()  	
