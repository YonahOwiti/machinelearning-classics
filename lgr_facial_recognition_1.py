'''
Created on Aug 18, 2017

@author: Varela

motive: Applies multiclass forecast
				
'''

import numpy as np 
import matplotlib.pyplot as plt 

from utils import get_facialexpression, error_rate, cross_entropy,  forward   
from utils_data import *

class FacialRecognizer1:
	#logistic regression model applied for multiclass classification
	def __init__(self, X, T):
		T, labels = class2numeric(T)

		Xtrain, Ttrain, Xvalid, Tvalid = split2test(X, T, perc=0.1, balanced=True)
		

		#Meta data
		self.labels 		= labels 
		self.K 				= len(labels) 
		self.N, self.D  = Xtrain.shape 


		#Input parameters
		self.Xtrain = Xtrain
		self.Ttrain = Ttrain
		self.Xvalid  = Xvalid
		self.Tvalid  = Tvalid

		#Output parameters
		self.W  			 = []
		


	
	def predict(self, X):
		if not(self.models):
			raise ValueError('model not fitted run fit!')

		N = len(X)
		XX 	= np.concatenate((np.ones((N,1)), X), axis=1) 
		YY  = np.zeros((N,), dtype=np.int32)
		Y   = np.zeros((K,), dtype=np.float32)


		#makes the predictions by combining 
		#we can do better but its the first approach
		for i in xrange(N):
			for j, l in enumerate(self.labels):
				Y[j] = forward(self.W[j], XX[i,:])

			YY[i]	= np.argmax(Y[j])	

		return YY 	

		
			
	def fit(self, max_iterations=1500, learning_rate=5e-8, verbose=True):	
	
		Xtrain = self.Xtrain
		Ttrain = self.Ttrain
		Xvalid = self.Xvalid
		Tvalid = self.Tvalid 

		N, D   = self.N, self.D 		
		Xtrain 		= np.concatenate((np.ones((N,1)), Xtrain), axis=1, ) 
		Xvalid 		= np.concatenate((np.ones((len(Tvalid),1)), Xvalid), axis=1, ) 


		self.train_costs   = np.zeros((max_iterations, self.K), dtype=np.float32)
		self.train_errors  = np.zeros((max_iterations, self.K), dtype=np.float32)
		# self.Ytrain = forward(W, Xtrain)    = np.zeros((max_iterations, self.K), dtype=np.float32)
		
		self.valid_errors   = np.zeros((max_iterations, self.K), dtype=np.float32)
		self.valid_costs   = np.zeros((max_iterations, self.K), dtype=np.float32)


		for j, l in enumerate(self.labels):
			if verbose: 
				print "Fitting %s - %d th class" % (l,j)

			# params
			# lr = 5e-7
			# max_iteration=150
			W  		= np.random.randn(D+1) / np.sqrt(D+1)
			
			for i in xrange(max_iterations):

				Ytrain = forward(W, Xtrain)
				Yvalid = forward(W, Xvalid)

				

				self.train_costs[i,j]  =  cross_entropy(Ttrain,Ytrain)
				self.train_errors[i,j] = error_rate(Ttrain,Ytrain)

				self.valid_costs[i,j]  = cross_entropy(Tvalid,Yvalid)
				self.valid_errors[i,j] =    error_rate(Tvalid,Yvalid)				

				W += learning_rate*Xtrain.T.dot(Ttrain-Ytrain)


				if verbose:
					if i % 5 == 0:
						print "klass=%s\ti=%d\tcost=%.3f\terror=%.3f\tset=training" % \
							(l,i,self.train_costs[i,j], self.train_errors[i,j])

						print "klass=%s\ti=%d\tcost=%.3f\terror=%.3f\tset=validating"  % \
							(l,i,self.valid_costs[i,j], self.valid_errors[i,j])

			self.W.append(W)				



def main():
	print 'Logistic regression model applied for multiclass classification >>'
	print 'Loading images this may take a while...'

	X, T = get_facialexpression(balance_ones=1)

	print 'Ready initializing data...'
	# Xtrain, Ttrain, XX, TT = split2test(X, T, perc=0.1, balanced=True)

	# TT, labels = class2numeric(TT)
	recognizer = FacialRecognizer1(X, T)
	recognizer.fit(max_iterations=150, learning_rate=5e-7, verbose=True)

	print 'Predicting...'
	Y = recognizer.predict(X)
	print 'Error rate:', error_rate(T, Y)





if __name__ == '__main__':
	main()
