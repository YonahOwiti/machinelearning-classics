'''
Created on Aug 17, 2017

@author: Varela

motive: Logistic regression for iris dataset
'''

import numpy as np 
import matplotlib.pyplot as plt 

from utils import get_facialexpression

def main():
	label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
	X, Y = get_facialexpression(balance_ones=False)

	while True: 
		for i in xrange(7):
			x, y = X[Y==i], Y[Y==i]	
			N = len(x)
			j = np.random.choice(N)
			plt.imshow(x[j].reshape(48,48), cmap='gray') # images have been flattened
			plt.title(label_map[y[j]])
			plt.show()
			prompt = raw_input('Quit? Enter Y:\n')
		
		if prompt == 'Y':
			break

if __name__ == '__main__': 
	main()	

