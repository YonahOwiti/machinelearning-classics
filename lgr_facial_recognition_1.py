'''
Created on Aug 19, 2017

@author: Varela

motive: Applies multiclass forecast / lazyprogrammer's implementation
course url: https://www.udemy.com/data-science-logistic-regression-in-python/learn/v4/t/lecture/5217332?start=0

'''


import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle 
from utils import sigmoid, error_rate, get_facialexpression, cross_entropy 
from utils_data import class1detect

class LogisticModel(object):
	
	def __init__(self):
		pass 

	def fit(self, X, Y, learning_rate=1e-6, reg=0, epochs=12000, show_figure=False):
		X, Y  = shuffle(X, Y)
		Xvalid, Yvalid = X[-1000:,:], Y[-1000:]
		X, Y  = X[:-1000,:], Y[:-1000]

		N, D  = X.shape
		self.W = np.random.randn(D) / np.sqrt(D)
		self.b = 0 

		costs = [] 
		best_validation_error = 1
		for i in xrange(epochs):
			pY = self.forward(X)
			# gradient descent step
			self.W -= learning_rate *(X.T.dot(pY-Y) + reg*self.W)
			self.b -= learning_rate *((pY-Y).sum() 	+ reg*self.b)

			if i % 20 == 0:
				pYvalid = self.forward(Xvalid)
				# c = sigmoid_cost(Yvalid, pYvalid)
				c = cross_entropy(Yvalid, pYvalid) 
				costs.append(c)	
				e = error_rate(Yvalid, pYvalid)
				print "i", i, "cost:", c, "error", e
				if e < best_validation_error:
					best_validation_error = e
		print "best_validation_error:", best_validation_error

		if show_figure:
			plt.plot(costs)
			plt.show()
	
	def forward(self, X):			
		return sigmoid(X.dot(self.W) + self.b)
  
	def predict(self, X):
		pY = self.forward(X)		
		return np.round(pY)

	def score(self, X, Y):
		prediction = self.predict(X)
		return 1 - error_rate(Y, prediction)

def main():
	X, Y = get_facialexpression(balance_ones=True)

	detect=1
	Y = class1detect(Y, detect)
	
	model = LogisticModel()
	model.fit(X, Y, show_figure=True)
	model.score(X, Y)



if __name__ == '__main__':
	main()  	
