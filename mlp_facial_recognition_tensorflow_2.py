'''
Created on Aug 25, 2017

@author: Varela

		motivation: Multilayer perceptron for facial expression recognition + tensorflow + ANY number of layers
			features: batching, momentum, decay, rmsprop, L2 regularization
	original url: https://github.com/lazyprogrammer/facial-expression-recognition/blob/master/ann_tf.py
'''

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle 

from utils import error_rate, get_facialexpression, y2indicator, softmax, init_weight_and_bias 
import tensorflow as tf 


class HiddenLayer(object):
	def __init__(self, M1, M2, an_id):
		self.id = an_id 
		self.M1 = M1 
		self.M2 = M2 
		W, b = init_weight_and_bias(M1, M2)
		self.W = tf.Variable(W.astype(np.float32), name='W%d' % an_id)
		self.b = tf.Variable(b.astype(np.float32), name='b%d' % an_id)
		self.params = [self.W, self.b] 

	def forward(self,X):
		return tf.nn.relu( tf.matmul(X, self.W) + self.b)


def AnnTensorflow2(object):
	def __init___(self, hidden_layer_sizes):
		self.hidden_layer_sizes = hidden_layer_sizes		
		return self 
		
	def fit(self, X, Y , learning_rate=10e-8, mu=0.99, decay=0.99, reg=10e-8,epochs=400, batch_sz=100, show_figure=False):
		X, Y = shuffle(X, Y)
		K = len(np.unique(Y))
		Y = y2indicator(Y, K).astype(np.float32)

		Xvalid, Yvalid = X[-1000:,:], Y[-1000:]
		Yvalid_flat = np.argmax(Yvalid, axis=1)
		Xtrain, Ytrain = X[:-1000,:], Y[:-1000]


		N, D = X.shape 
				

		#Build hidden layers
		M1 = D 
		self.hidden_layers = [] 
		self.params = [] 
		for an_id, M2 in enumerate(self.hidden_layer_sizes):
			h = HiddenLayer(M1, M2, an_id)
			self.hidden_layers.append(h)
			self.params += h.params 
			M1=M2
		
		M2=K
		an_id=len(self.hidden_layer_sizes)
		W , b =   init_weight_and_bias(M1, M2)
		self.W = tf.Variable(W.astype(np.float32), name='W%d' % an_id)
		self.b = tf.Variable(b.astype(np.float32), name='b%d' % an_id)
		
		self.params += [self.W, self.b]

		X = tf.placeholder(tf.float32,shape=(None,D),name='X')
		Y = tf.placeholder(tf.float32,shape=(None,K),name='Y')
		Yish = self.forward(X)

		# cost functions
		rcost = reg*tf.sum([tf.nn.l2_loss(p) for p in self.params]) # L2 regularization costs 
		cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Yish)) + rcost

		train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)
		predict_op = tf.argmax(Yish, 1)

		LL = [] 
		n_batches = int(N / batch_sz)
		best_validation_error=1
		init = tf.global_variables_initializer()
		with tf.Session() as session:
			session.run(init)
			
			for i in xrange(epochs):
				X, Y = shuffle(X, Y)

				for j in xrange(n_batches):
					Xbatch = Xtrain[j*(batch_sz):(j+1)*batch_sz,:]
					Ybatch = Ytrain[j*(batch_sz):(j+1)*batch_sz,:]

					session.run(
						train_op,
						feed_dict={
							X:Xbatch,
							Y:Ybatch
						}
					)

					if j % 100 == 0:
						pY = session.run(predict_op, feed_dict={X:Xvalid})
						c = session.run(cost, feed_dict={X:Xvalid, Y:Yvalid})
						err = error_rate(Yvalid_flat,pY)
						LL.append(c)
						print "i:%d\tj:%d\tnb:%d\tc:%.3f\te:%.3f\t" % (i,j,n_batches ,c,err)

					if err < best_validation_error:
						best_validation_error = err
			print "best_validation_error:", best_validation_error
		
		if show_figure:
			plt.plot(LL)
			plt.show()

	def forward(self, X):	
		Z = X 
		for h in self.hidden_layers:
			Z = h.forward(X)
		return tf.matmul(Z, self.W) + self.b
	
def main():
	X, Y = get_facialexpression(balance_ones=True)
	
	ann = AnnTensorflow2([2000, 1000, 500])
	print type(ann)
	ann.fit(X, Y, show_figure=True)
	



if __name__ == '__main__':
	main()  	
