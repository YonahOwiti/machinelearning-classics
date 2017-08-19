'''
Created on Aug 18, 2017

@author: Varela

motive: Applies multiclass forecast
				
'''

import numpy as np 

from utils import get_facialexpression, error_rate, cross_entropy,  forward   
from utils_data import *

class LogisticRegressionK(object):
	#Logistic Regression model applied for K classifications
	def __init__(self, X, T, labels=[], split=False):
		#X 			N,D 	sample, features 
		#T 			N,  	sample,		
		#labels K,  	np.array holding the strings for class names
		#split  bool  np.array if true reserves 10% of sample for validation

		if (isinstance(labels, list)) and not(labels):
			# lables is an empty list
			self.labels = np.unique(T)
		else:
			self.labels = labels 

		


		# T, labels = class2numeric(T)

		# Xtrain, Ttrain, Xvalid, Tvalid = split2test(X, T, perc=0.1, balanced=True)
		

		
		#Input parameters		
		self.split = split 
		if self.split:
			self.Xtrain, self.Ttrain, self.Xvalid, self.Tvalid = split2test(X, T, perc=0.1, balanced=True)			
		else: 	
			self.Xtrain, self.Ttrain = X, T 

		
		#Meta data				
		self.K 				  = len(self.labels) 	
		self.N, self.D  = self.Xtrain.shape 


		#Input parameters
		# self.X = X.astype(np.float32)
		# self.T = T.astype(np.int)
		# self.Xtrain = X.astype(np.float32)
		# self.Ttrain = T.astype(np.int)
		# self.Xvalid  = Xvalid.astype(np.float32)
		# self.Tvalid  = Tvalid.astype(np.int)

		#Output parameters
		self.W  			 = []
		


	
	def predict(self, X):
		if not(self.W):
			raise ValueError('model not fitted run fit!')

		N = len(X)
		XX 	= np.concatenate((np.ones((N,1)), X), axis=1) 
		YY  = np.zeros((N,), dtype=np.float32)
		Y   = np.zeros((self.K,), dtype=np.float32)


		#makes the predictions by combining 
		#we can do better but its the first approach
		for i in xrange(N):
			for j, l in enumerate(self.labels):
				Y[j] = forward(self.W[j], XX[i,:])

			YY[i]	= np.argmax(Y[j])	

		return YY 	

		
			
	def fit(self, max_iterations=1500, learning_rate=5e-8, verbose=True):	
	
		# X = self.X
		# T = self.T
		Xtrain = self.Xtrain
		Ttrain = self.Ttrain
		# Xvalid = self.Xvalid
		# Tvalid = self.Tvalid 

		N, D   = self.N, self.D 		
		# X 		= np.concatenate((np.ones((N,1)), X), axis=1, ) 
		Xtrain 		= np.concatenate((np.ones((N,1)), Xtrain), axis=1, ) 
		# Xvalid 		= np.concatenate((np.ones((len(Tvalid),1)), Xvalid), axis=1, ) 


		self.train_costs   = np.zeros((max_iterations, self.K), dtype=np.float32)
		self.train_errors  = np.zeros((max_iterations, self.K), dtype=np.float32)
		# self.Ytrain = forward(W, Xtrain)    = np.zeros((max_iterations, self.K), dtype=np.float32)
		if self.split:
			n = len(self.Xvalid)
			self.Xvalid = np.concatenate((np.ones((n,1)), self.Xvalid), axis=1, ) 
			self.valid_errors   = np.zeros((max_iterations, self.K), dtype=np.float32)
			self.valid_costs   = np.zeros((max_iterations, self.K), dtype=np.float32)


		for j, l in enumerate(self.labels):

			T1 = class1detect( Ttrain,j )
			if self.split:
				T1valid = class1detect( self.Tvalid,j )
			if verbose: 
				print "Fitting %s - %d th class" % (l,j)

			# params
			# lr = 5e-7
			# max_iteration=150
			W  		= np.random.randn(D+1) / np.sqrt(D+1)
			
			for i in xrange(max_iterations):

				Ytrain = forward(W, Xtrain)
				# Ytrain = forward(W, Xtrain)
				

				
				self.train_costs[i,j]  =  cross_entropy(T1,Ytrain)
				self.train_errors[i,j] =  error_rate(T1,Ytrain)
				# self.train_costs[i,j]  =  cross_entropy(Ttrain,Ytrain)
				# self.train_errors[i,j] =  error_rate(Ttrain,Ytrain)
				if self.split:
					Yvalid = forward(W, self.Xvalid)
					self.valid_costs[i,j]  = cross_entropy(T1valid,Yvalid)
					self.valid_errors[i,j] =    error_rate(T1valid,Yvalid)				

				# W -= learning_rate*X.T.dot(Y-T)
				W += learning_rate*Xtrain.T.dot(T1-Ytrain)


				if verbose:
					if i % 5 == 0:
						if self.split: 
							print "klass=%s\ti=%d\tcost=%.0f/%.0f\terror=%.3f/%.3f\tset=(training/validation)" % \
								(l,i,self.train_costs[i,j], self.valid_costs[i,j], self.train_errors[i,j] , self.valid_errors[i,j])
						else:		
							print "klass=%s\ti=%d\tcost=%.0f\terror=%.3f" % \
								(l,i,self.train_costs[i,j],  self.train_errors[i,j] )

						

			self.W.append(W)				



def main():
	print 'Logistic regression model applied for multiclass classification >>'
	print 'Loading images this may take a while...'

	X, T = get_facialexpression(balance_ones=1)
	T, labels = class2numeric(T)
	# T 	= class1detect(T,1)

	print 'Ready initializing data...'
	# Xtrain, Ttrain, XX, TT = split2test(X, T, perc=0.1, balanced=True)

	# TT, labels = class2numeric(TT)
	recognizer = LogisticRegressionK(X, T, labels, split=True)
	recognizer.fit(max_iterations=350, learning_rate=5e-7, verbose=True)

	print 'Predicting...'
	Y = recognizer.predict(X)
	print 'Error rate:', error_rate(T, Y)





if __name__ == '__main__':
	main()
