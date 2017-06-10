# Kevin 
# 2017.06.10
# ================================================================
"""An implement of Bidirectional Recurrent Neural Network(LSTM) with MNIST data using Tensorflow"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets('/tmp/MNIST_data/', one_hot=True)

# Basic model parameters
learning_rate = 1e-4
training_epoch = 1000
batch_size = 100
display_step = 1

# Network Parameters
n_input = 28 # MNIST data input ( img shape: 28*28 )
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes ( 0-9 digits )


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


# Define weights & biases
weights = {
    'out':tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}

biases = {
    'out':tf.Variable(tf.random_normal([n_classes]))
}

def BiRNN(x, weights, biases):
    # Unstack to get a list of 'n_steps' tensor of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)
    
    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    
    # Get lstm cell output
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    
    return tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'])


logits = BiRNN(x, weights, biases)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Evaluation
correct_prediction = tf.equal(tf.arg_max(logits, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()

sess.run(init)


with tf.device('/cpu:0'):
    training_images = mnist.train.images.reshape([mnist.train.num_examples, n_steps, n_input])
    training_labels = mnist.train.labels
    testing_images = mnist.test.images.reshape([mnist.test.num_examples, n_steps, n_input])
    testing_labels = mnist.test.labels

# training & testing & displaying
for epoch in range(training_epoch):
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_input])
        
        sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})
    if epoch % display_step == 0:
        print('Epoch: %d, training loss: %.4f, training accuracy: %.4f,testing loss: %.4f, testing accuracy: %.4f'
              %(epoch+1, 
                sess.run(loss, feed_dict={x:training_images, y:training_labels}),
                sess.run(accuracy, feed_dict={x:training_images, y:training_labels}),
                sess.run(loss, feed_dict={x:testing_images, y:testing_labels}),
                sess.run(accuracy, feed_dict={x:testing_images, y:testing_labels})))
