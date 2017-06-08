# Kevin
# 2017.06.08
# ========================================================
"""Linear regression with tensorflow"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Model parameters
training_step = 200
display_step = 20

# Regression coefficient
W_true = [[1],[2]]
bias_true = [0.5]

# Generate data
x_data = np.float32(np.random.randn(100, 2))
y_data = np.dot(x_data, W_true) + bias_true

# Create a linear model
W = tf.Variable(tf.random_uniform([2,1], minval=-1.0, maxval=1.0))
bias = tf.Variable(tf.zeros([1]))
y = tf.add(tf.matmul(x_data, W), bias)

# Calculate loss(optimization object)
loss = tf.reduce_mean(tf.square(y - y_data))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# Initialize variable
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)

for step in range(training_step):
    sess.run(optimizer)
    if (step + 1) % display_step == 0:
        print('step: %d, W:%s, bias:%s' %(step+1, sess.run(W), sess.run(bias)))
