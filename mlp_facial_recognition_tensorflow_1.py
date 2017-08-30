'''
Created on Aug 29, 2017

@author: Varela

motivation: Multilayer perceptron for facial expression recognition + tensorflow
	
'''
import numpy as np 
import tensorflow as tf 

import matplotlib.pyplot as plt 
from utils import get_facialexpression, init_weight_and_bias , error_rate, y2indicator 
from sklearn.utils import shuffle 

class AnnTensorflow1(object):
	
	def __init__(self, M):
		self.M = M 

	def fit(self, X, Y, learning_rate=10e-5, epochs=200, reg=10e-8, batch_sz=200, show_fig=False, activation=tf.tanh):
		X, Y = shuffle(X, Y)
		K = len(np.unique(Y))  

		T = y2indicator(Y, K).astype(np.float32)
		Xvalid, Yvalid, Tvalid = X[-1000:,], Y[-1000:], T[-1000:,:] 
		Xtrain, Ytrain, Ttrain = X[:-1000,:], Y[:-1000],T[:-1000,:] 

		N, D = Xtrain.shape
		

		#Varianel initialization
		W1, b1 = init_weight_and_bias(D,self.M)
		W2, b2 = init_weight_and_bias(self.M,K)



		self.W1 = tf.Variable(W1.astype(np.float32), 'W1')
		self.b1 = tf.Variable(b1.astype(np.float32), 'b1')
		self.W2 = tf.Variable(W2.astype(np.float32), 'W2')
		self.b2 = tf.Variable(b2.astype(np.float32), 'b2')
		self.params = [self.W1, self.b1, self.W2, self.b2] 
		# Define placeholders
		X = tf.placeholder(tf.float32,shape=(None,D),name='X')
		T = tf.placeholder(tf.float32,shape=(None,K),name='Y')

		
		

		Z = activation(tf.matmul(X, self.W1) + self.b1) 		
		Yish = tf.matmul(Z, self.W2) + self.b2 

		rcost  = reg*tf.reduce_sum([tf.nn.l2_loss(p) for p in self.params])
		cost   = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits(labels=T, logits=Yish) ) + rcost 
		
		
		train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
		self.predict_op = tf.argmax(Yish, 1)

		n_batches = N // batch_sz 
		costs=[] 
		errors=[] 
		init = tf.global_variables_initializer()
		with tf.Session() as session:
			session.run(init)

			for i in xrange(epochs):
				Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
				for j in xrange(n_batches): 
					Xbatch = Xtrain[j*batch_sz:(j+1)*batch_sz,:]
					Ybatch = Ytrain[j*batch_sz:(j+1)*batch_sz]
					Tbatch = Ttrain[j*batch_sz:(j+1)*batch_sz,:]

					session.run(train_op,
						feed_dict={
							X: Xbatch,
							T: Tbatch 
					})

					if j % 10 == 0: 
						c = session.run(cost, feed_dict={X:Xvalid, T:Tvalid} )
						pYvalid  = session.run( self.predict_op, feed_dict={X: Xvalid} )
						err = error_rate(Yvalid, pYvalid)
						print "i:%d\tj:%d\tc:%.3f\terr:%.3f\t" % (i,j,c,err)	
						costs.append(c)
						errors.append(err)

		if show_fig:
			plt.title('costs')
			plt.plot(costs)
			plt.show()

			plt.title('error rate')
			plt.plot(errors)
			plt.show()
				

def main():
	X, Y = get_facialexpression(balance_ones=True)
	
	M = 5000 	
	ann = AnnTensorflow1(M)
	ann.fit(X, Y, learning_rate=10e-7, reg=10e-6, show_fig=True)




if __name__ == '__main__':
	main()  	





