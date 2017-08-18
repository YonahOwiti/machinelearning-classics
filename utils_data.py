'''
Created on Aug 18, 2017

@author: Varela

motive: Provides datareshape such as sampling, balanced sampling, convert to classes to numeral
'''

from sklearn.utils import shuffle
import numpy as np 

def class1detect(T, detect): 
	mask = T == detect
	T[mask] = 1
	T[~mask] = 0
	return T 

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

def split2test(X, Y, perc=0.1, balanced=False):
	# Splits the randomized samples X, Y into Xtrain, Xtest, Ytrain, Ytest
	X, Y= shuffle(X,Y)
	
	N = len(Y)
	n = int(N * perc)

	index = np.arange(N)
	# handles balanced class
	if balanced:
		klasses = np.unique(Y)
		k = len(klasses)
		r = int(n / k)
		z = r - n*k 
		testindex = np.zeros((n,), dtype=np.int32)
		for i, klass in enumerate(klasses):
			m = r 
			if z > 0:
				m+=1
				z-=1

			klassindex = np.random.choice(index[Y==klass], size=m, replace=False)	
			testindex[i*m:(i+1)*m] = klassindex		

	else:
		testindex = np.random.choice(index, size=n, replace=False)

	trainindex = np.setdiff1d(index,testindex,assume_unique=True)	

	Xtrain 	= X[trainindex,:]
	Ytrain 	= Y[trainindex]
	Xtest 	= X[testindex,:]
	Ytest 	= Y[testindex]

	return Xtrain, Ytrain, Xtest, Ytest

