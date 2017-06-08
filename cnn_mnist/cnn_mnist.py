# Kevin
# 2017.06.08
# =========================================================
"""CNN with tensorflow"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Basic model paramters
training_step = 100000
display_step = 100
batch_size = 50

learning_rate = 1e-4

# Import MNIST data
mnist = input_data.read_data_sets('/tmp/MNIST_data/',one_hot=True)

# Net input
x = tf.placeholder('float', shape=[None, 784])
y = tf.placeholder('float', shape=[None, 10])
keep_prob = tf.placeholder('float')

# Basic function
def weight_variable(shape):
    
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)    

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def avg_pool(x):
    return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# Net cofficients
W = {'conv1':weight_variable([5, 5, 1, 32]),
     'conv2':weight_variable([5, 5, 32, 64]),
     'fc1':weight_variable([7*7*64, 1024]),
     'fc2':weight_variable([1024,10])}

bias = {'conv1':bias_variable([32]),
        'conv2':bias_variable([64]),
        'fc1':bias_variable([1024]),
        'fc2':bias_variable([10])}

def convolution(x,W,bias):
    x = tf.reshape(x, [-1, 28, 28, 1])
    
    # convolution
    conv1 = conv2d(x, W['conv1']) + bias['conv1']
    # active
    conv1 = tf.nn.relu(conv1)
    # max poll
    pool1 = max_pool(conv1)
    
    # convolution
    conv2 = conv2d(pool1, W['conv2']) + bias['conv2']
    # active
    conv2 = tf.nn.relu(conv2)
    # max pool
    pool2 = max_pool(conv2)
    
    # reshape pool2 to fit full connection layer
    pool2 = tf.reshape(pool2, [-1, 7*7*64])
    # full connect layer1
    fc1 = tf.nn.relu(tf.matmul(pool2, W['fc1']) + bias['fc1'])
    
    # dropout
    
    fc1_drop = tf.nn.dropout(fc1, keep_prob)
    
    # output layer
    logits = tf.nn.softmax(tf.matmul(fc1_drop, W['fc2'])+bias['fc2'])
    
    
    return logits



logits = convolution(x,W,bias)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)


for step in range(training_step):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.5})
    
    if (step+1) % display_step == 0:
        print('step: %d, training batch accuracy: %.4f' 
              %(step+1, sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0})))
        
print('Optimization finish!')

test_accuracy = []
test_epoch = int(mnist.test.num_examples / batch_size)

for step in range(test_epoch):
    batch_xs, batch_ys = mnist.test.next_batch(batch_size)
    test_accuracy.append(sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0}))    
test_accuracy = tf.reduce_mean(test_accuracy)

print('Testing accuracy: %.4f' %sess.run(test_accuracy))



