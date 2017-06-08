# Kevin
# 2017.06.08
# ==========================================
""" Artificial neural network with MNIST using Tensorflow."""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/MNIST_data/', one_hot=True)


# Basic model paramters
learning_rate = 0.01
training_epochs = 20
batch_size = 100
display_step = 1

# Network parameters
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256
n_class = 10

# Store layers weights & bias
weights = {
    'h1':tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2, n_class]))}

biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_class]))
}

# tf Graph input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


# Greate model
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
    logits = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return logits

logits = multilayer_perceptron(x, weights, biases)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialization
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()

sess.run(init)


correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(logits,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    for step in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost = sess.run([optimizer, loss], 
                           feed_dict={x:batch_xs, y:batch_ys})
        
        avg_cost += cost/total_batch
    
    if epoch % display_step == 0:
        print('Epoch:%d, cost: %.4f, training accuracy: %.4f'
              %(epoch+1, avg_cost, sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys})))
    
print('Optimize finish!')
print('Test accuracy: %.4f'
      %(sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})))
