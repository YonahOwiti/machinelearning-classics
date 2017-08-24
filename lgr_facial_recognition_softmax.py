'''
Created on Aug 24, 2017

@author: Varela

motive: LogisticRegression + softmax
course url: https://www.udemy.com/data-science-deep-learning-in-python/learn/v4/t/lecture/5239240?start=0

'''



import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle 
from utils import sigmoid, error_rate, get_facialexpression, cross_entropy, y2indicator 


class LogisticModel(object):
	
	def __init__(self):
		pass 

	def fit(self, X, Y, learning_rate=1e-6, reg=0, epochs=12000, show_figure=False):
		
		X, Y  = shuffle(X, Y)
		N, D  = X.shape
		K = len(set(Y))
		T 		= y2indicator(Y,K)

		Xvalid, Yvalid, Tvalid = X[-1000:,:], Y[-1000:], T[-1000:,:] 
		X, Y, T  = X[:-1000,:], Y[:-1000], T[:-1000,:] 


		
		self.W = np.random.randn(D,K) / np.sqrt(D +K)
		self.b = np.zeros((K,))


		costs = [] 
		best_validation_error = 1
		for i in xrange(epochs):
			pY = self.forward(X)
			# gradient descent step
			self.W -= learning_rate *(X.T.dot(pY-Y) + reg*self.W)
			self.b -= learning_rate *((pY-Y).sum(axis=0) 	+ reg*self.b)

			if i % 10 == 0:
				pYvalid = self.forward(Xvalid)
				
				c = cross_entropy(Yvalid, pYvalid) 
				costs.append(c)	
				e = error_rate(Yvalid, np.argmax(pYvalid,axis=1))

				print "i", i, "cost:", c, "error", e
				if e < best_validation_error:
					best_validation_error = e
		print "best_validation_error:", best_validation_error

		if show_figure:
			plt.plot(costs)
			plt.show()
	
	def forward(self, X):			
		return softmax(X.dot(self.W) + self.b)
  
	def predict(self, X):
		pY = self.forward(X)		
		return np.argmax(pY, axis=1)

	def score(self, X, Y):
		prediction = self.predict(X)
		return 1 - error_rate(Y, prediction)

def main():
	X, Y = get_facialexpression(balance_ones=True)
	
	model = LogisticModel()
	model.fit(X, Y, show_figure=True)
	print model.score(X, Y)



if __name__ == '__main__':
	main()  	
