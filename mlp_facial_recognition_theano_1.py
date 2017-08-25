'''
Created on Aug 24, 2017

@author: Varela

motivation: Multilayer perceptron for facial expression recognition + theano
	
'''

import numpy as np 

import theano
import theano.tensor as T 

from sklearn.utils import shuffle 
from utils import get_facialexpression, y2indicator, sigmoid, softmax, classification_rate , error_rate 

class AnnTheano(object):
	def __init__(self, M):
		self.M = M 

	def fit(self, Xin, Yin, learning_rate=10e-7, reg=10e-8, epochs=10000, show_figure=False):
		Nvalid = 500
		N, D  = Xin.shape 
		K =  len(np.unique(Yin))
		Xin, Yin  = shuffle(Xin, Yin)

		Xtrain, Ytrain = Xin[-Nvalid:,:],  Yin[-Nvalid:,]
		Xvalid, Yvalid = Xin[:-Nvalid,:], Yin[:-Nvalid,]
		Ttrain, Tvalid = y2indicator(Ytrain, K),  y2indicator(Yvalid, K)

		#Initialize Wi,bi
		W1_init = np.random.randn(D,self.M) / np.sqrt(D + self.M)
		b1_init = np.random.randn(self.M)  / np.sqrt(self.M)
		W2_init = np.random.randn(self.M,K) / np.sqrt(K + self.M)
		b2_init = np.random.randn(K)  / np.sqrt(K)

		#Theano shared
		W1 = theano.shared(W1_init, 'W1')
		b1 = theano.shared(b1_init, 'b1')
		W2 = theano.shared(W2_init, 'W2')
		b2 = theano.shared(b2_init, 'b2')

		#Theano variables 
		thX = T.matrix('X')
		thT = T.matrix('T')
		thZ = sigmoid(thX.dot(W1) + b1)
		thY = T.nnet.softmax( thZ.dot(W2) + b2)


		#Theano updatebles
		costs = -(thT * np.log(thY) + (1 - thT)*np.log((1 - thY))).sum() 
		prediction = T.argmax(thY, axis=1)

		W1_update = W1 - learning_rate * ( T.grad(costs, W1) + reg*W1)
		b1_update = b1 - learning_rate * ( T.grad(costs, b1) + reg*b1)

		W2_update = W2 - learning_rate * ( T.grad(costs, W2) + reg*W2)
		b2_update = b2 - learning_rate * ( T.grad(costs, b2) + reg*b2)
		

		self._train = theano.function(
			inputs = [thX, thT],
			updates= [(W1, W1_update), (b1, b1_update), (W2, W2_update), (b2, b2_update)],
		)

		self._predict = theano.function(
			inputs = [thX, thT],
			outputs=[costs, prediction],
		)

		
		train_costs = [] 
		train_errors=[]
		valid_costs = []
		valid_errors = [] 

		for i in xrange(epochs):
			self._train(Xtrain,Ttrain)
			if i % 10 == 0:
				ctrain, pYtrain = self._predict(Xtrain,Ttrain)
				err = error_rate(Ttrain, pYtrain)
				train_costs.append(ctrain)
				train_errors.append(err)

				cvalid, pYvalid = self._predict(Xvalid,Tvalid)
				err = error_rate(Tvalid, pYvalid)
				valid_costs.append(cvalid)
				valid_errors.append(err)
				print "i=%d\tc=%.3f\terr==%.3f\t" % (i,cvalid,err)

		cvalid, pYvalid = self._predict(Xvalid,Tvalid)
		err = error_rate(Tvalid, pYvalid)
		valid_costs.append(cvalid)
		valid_errors.append(err)		

		print "i=%d\tc=%.3f\terr==%.3f\t" % (epochs,cvalid,err)

		
		print "Final train classification rate", classification_rate(Ytrain, pYtrain)
		print "Final valid classification rate", classification_rate(Yalid,  pYalid)
			
		plt.title('Multi layer perceptron: Costs')
		plt.xlabel('iterations')
		plt.ylabel('costs')
		legend1, = plt.plot(train_costs, label='train cost')
		legend2, = plt.plot(valid_costs, label='valid cost')
		plt.legend([legend1, legend2,])
		plt.show()


		plt.title('Multi layer perceptron: Error rates')
		plt.xlabel('iterations')
		plt.ylabel('error rates')
		legend1, = plt.plot(train_errors, label='train error')
		legend2, = plt.plot(valid_errors, label='valid error')
		plt.legend([legend1, legend2,])
		plt.show()		

		

	def predict(self, X):
		_, pY = self._predict(X)
		return pY
		

	def score(self, X, Y):
		pY = predict(X)
		return classification_rate(Y, pY)
	


def main():
	X, Y =  get_facialexpression(balance_ones=True)

	M = 2000
	ann = AnnTheano(M)
	ann.fit(X, Y)
	print "score:", ann.score(X,Y)

if __name__ == '__main__':
	main()