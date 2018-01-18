import collections
import random
import numpy as np 
import time 

import tensorflow as tf
from tensorflow.contrib import rnn 

start_time= time.time() 
def elapsed(sec):
    if sec<60:
        return str(sec) + ' sec'
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"
        

text = 'long ago , the mice had a general council to consider what measures they could take to outwit their common enemy , the cat . some said this , and some said that but at last a young mouse got up and said he had a proposal to make , which he thought would meet the case . you will all agree , said he , that our chief danger consists in the sly and treacherous manner in which the enemy approaches us . now , if we could receive some signal of her approach , we could easily escape from her . i venture , therefore , to propose that a small bell be procured , and attached by a ribbon round the neck of the cat . by this means we should always know when she was about , and could easily retire while she was in the neighbourhood . this proposal met with general applause , until an old mouse got up and said that is all very well , but who is to bell the cat ? the mice looked at one another and nobody spoke . then the old mouse said it is easy to propose impossible remedies .'

def build_dataset(words):
  count= collections.Counter(words).most_common()
  dictionary= dict()
  for word, _ in count:
      dictionary[word]= len(dictionary)
  reverse_dictionary= dict(zip(dictionary.values(), dictionary.keys()))
  return dictionary, reverse_dictionary

words = text.split()
word2idx, idx2word = build_dataset(words)
output = random.sample(list(word2idx.items()),k=5)
print(output)    

words = text.split()
word2idx, idx2word = build_dataset(words)
output = random.sample(list(word2idx.items()),k=5)
print(output)

# Parameters
vocab_size= len(words)
n_input=3 

learning_rate=0.001
training_iters=50000
display_step= 1000
n_input=3 

#number of units in RNN cell
n_hidden= 512

#RNN output node weights and biases
weights={
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases= {
    'out': tf.Variable(tf.random_normal([vocab_size]))    
}

#tf Graph input
x = tf.placeholder(tf.float32, shape=(None, n_input,1 ), name='x')
y = tf.placeholder(tf.float32, shape=(None, vocab_size), name='y')

def cell():
    #1-layer LSTM with n_hidden units.
    return rnn.core_rnn_cell.BasicLSTMCell(n_hidden)

def RNN(x, weights, biases):
    
  #reshape to [1, n_input]
  x = tf.reshape(x, [-1, n_input])
  
  #Generate a n_input-element sequence of inputs
  #(eg. [had] [a] [general] ->  [20] [6] [33])
  x = tf.split(x, n_input, 1)
  

  rnn_cell=cell()
  #generate prediction 
  lstm_cell= rnn.MultiRNNCell([rnn_cell])
  outputs, states= rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
  
  #there are n_input outputs but
  #we only want the last output
  return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred= tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#Initializing the variables
init= tf.global_variables_initializer()    

#Launch the graph
with tf.Session() as session:
	session.run(init)
	step=0 
	offset= random.randint(0, n_input)
	end_offset= n_input+1 
	acc_total=0 
	loss_total=0 
  
	while (step < training_iters):
		#Generate a minibatch. Add some randomness
		if offset> (len(words)-end_offset):
		    offset=random.randint(0,n_input+1)



		symbols_in_keys= [[word2idx[words[i]]] for i in range(offset, offset+n_input)]   
		symbols_in_keys= np.reshape( np.array(symbols_in_keys), (-1, n_input,1) )

		symbols_out_onehot= np.zeros((vocab_size), dtype=np.float32)
		symbols_out_onehot[word2idx[words[offset+n_input]]]=1.0
		symbols_out_onehot=np.reshape(symbols_out_onehot,[1,-1])

		_, acc, loss, one_hotpred= session.run(
		    [optimizer, accuracy, cost, pred],
		    feed_dict={x: symbols_in_keys, y:symbols_out_onehot}            
		)    
		loss_total += loss 
		acc_total  += acc 

		if (step+1) % display_step==0:
			msg= 'Iter=' + str(step+1) 
			msg+=', Average Loss=' + "{:.6f}".format(loss_total/display_step) 
			msg+= ', Average Accuracy=' + "{:.2f}".format(100*acc_total/display_step)
			print(msg)
			acc_total=0
			loss_total=0
			symbols_in= [words[i] for i in range(offset, offset + n_input)]
			symbols_out= words[offset + n_input]
			symbols_out_pred= idx2word[int(tf.argmax(one_hotpred, 1).eval())]

			print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))

		step+=1
		offset +=(n_input+1)
			  
print('optimization finished')
print('Elapsed time:', elapsed(time.time()-start_time))
