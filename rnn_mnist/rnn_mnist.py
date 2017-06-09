# Kevin
# 2017.06.09
# ========================================================
"""An implement of Recurrent neural network with MNSIT data using Tensorflow."""

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/MNIST_data/', one_hot=True)

# Basic model Parameters
learning_rate = 1e-4
training_epoch = 100
batch_size = 100
display_epoch = 1

# Network Parameters
n_input = 28 # MNIST data input ( img shape: 28 * 28 )
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    'out':tf.Variable(tf.random_normal([n_hidden, n_classes]))
}

biases = {
    'out':tf.Variable(tf.random_normal([n_classes]))
}

# Build RNN netword
def RNN(x, weights, biases):
    
    # Unstack to get a list of 'n_class' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)
    
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    
    # Get a lstm output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
    


logits = RNN(x, weights, biases)


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# Evaluation
correct_prediction = tf.equal(tf.arg_max(logits, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize and start a session of Tensorflow
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)


# Training and dispalying
for epoch in range(training_epoch):
    training_step = int(mnist.train.num_examples/batch_size)
    for step in range(training_step):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        
        batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
        
        sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})
        
    if epoch % display_epoch == 0:
        xs,ys = mnist.train.images, mnist.train.labels
        xs = xs.reshape(mnist.train.num_examples, n_steps, n_input)
        loss, acc = sess.run([cost, accuracy], 
                             feed_dict={x:xs, y:ys})
        print('Training epoch: %d, Training loss: %.4f, Training accuracy: %.4f'
              %(epoch+1, loss, acc))

print('Optimize finish!')


# Evaluate in testing data
xs, ys = mnist.test.images, mnist.test.labels
xs = xs.reshape((mnist.test.num_examples, n_steps, n_input))
print('Testing accuracy: %.4f' 
      %(sess.run(accuracy, feed_dict={x:xs, y:ys})))
