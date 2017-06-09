# Kevin
# 2017.06.09
# =================================================
"""An implement of sparse autoencoder with MNIST using Tensorflow"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/MNIST_data/', one_hot = True)

# Parameters
learning_rate = 1e-2
training_epochs = 5
batch_size = 200
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784

x = tf.placeholder(tf.float32, [None, n_input])

weights = {
    'encoder_h1':tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1':tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2':tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases = {
    'encoder_h1':tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_h1':tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_h2':tf.Variable(tf.random_normal([n_input]))
}

# Building the encoder
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), 
                                   biases['encoder_h1']))
    
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), 
                                   biases['encoder_h2']))
    return layer_2
    
def decoder(x):
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                  biases['decoder_h1']))
    
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h2']),
                                  biases['decoder_h2']))
    return layer_4

# Construct model
encoder_op = encoder(x)
decoder_op = decoder(encoder_op)

 # Prediction
logits = decoder_op

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(x - logits))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initialzer
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()

sess.run(init)


# Training 
for epoch in range(training_epochs):
    training_step = int(mnist.train.num_examples/batch_size)     
    for step in range(training_step):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x:batch_xs})
            
    # Display logs per epoch step
    if epoch % display_step == 0:
        print('Epoch: %d, cost: %.4f' %(epoch+1, sess.run(cost, feed_dict={x:mnist.train.images})))
        
print('Optimization finish!')

# Applying encoder and decoder over test examples
encode_decode = sess.run(logits, feed_dict={x:mnist.test.images[:examples_to_show]})


# Compare original images with their reconstructions
f, a = plt.subplots(2, 10, figsize=(10,2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28,28)))
f.show()
plt.draw()
plt.show()


