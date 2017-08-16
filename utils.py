'''
Created on Aug 16, 2017

@author: Varela

'''

import numpy as np 
import pandas as pd 

def get_iris(detect='Iris-setosa'):
	X = []; Y =[]
	if not( detect in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']): 
		raise ValueError( 'value to detect - %s invalid' % (detect))
	df = pd.read_csv('./datasets/iris/iris.data', sep=',')
	X=  df.values[:,:-1]
	Y=  df.values[:,-1] == detect

	return X, Y 
