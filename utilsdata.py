'''
Created on Aug 18, 2017

@author: Varela

motive: Provides datareshape such as sampling, balanced sampling, convert to classes to numeral
'''

from sklearn.utils import shuffle
import numpy as np 

def class2numeric(Y):
	first=True 
	YY = np.zeros(Y.shape,dtype=np.int32)
	klass = np.unique(Y)
	for i, y in enumerate(klass):
		if not(first):
			YY[Y==y]=i 
		else:
			first=False 
	return YY, klass  		

def split2test(X, Y, perc=0.1):
	# Splits the randomized samples X, Y into Xtrain, Xtest, Ytrain, Ytest
	X, Y= shuffle(X,Y)
	
	N = len(Y)
	n = int(N * perc)

	index = np.arange(N)
	testindex = np.random.choice(np.arange(N), size=n, replace=False)

	masktrain = np.where(testindex in index)
	masktest  = np.where(not(testindex in index))

	Xtrain 	= X[masktrain,:]
	Ytrain 	= Y[masktrain]
	Xtest 	= X[masktest,:]
	Ytest 	= Y[masktest]

	return Xtrain, Ytrain, Xtest, Ytest



	
	

