# Kevin
# 2017.06.08

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Model parameters
training_step = 1000
display_step = 100
batch_size = 100
learning_rate = 0.01

# Import MNIST data 
mnist = input_data.read_data_sets('/tmp/MNIST_data/', one_hot=True)

## Create a softmax regression model
# Model input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Model coefficients
with tf.device('/cpu:0'):
    W = tf.Variable(tf.random_normal([784, 10]))
    bias = tf.Variable(tf.zeros([10]))

# predict
logits = tf.add(tf.matmul(x, W), bias)

# loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# initializer
init = tf.global_variables_initializer()

# evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(init)

for step in range(training_step):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})
    
    if (step+1) % display_step == 0:
        print('step: %s, training accuracy: %.4f'
              %(step+1, sess.run(accuracy,feed_dict={x:batch_xs, y:batch_ys})))
    
    
print('Optimization finish!')
print('Testing accuracy: %.4f'%sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels}))
        

