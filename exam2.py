__author__ = 'jaehyek'

import tensorflow as tf
import numpy as np

# x_data = 2 x 100
x_data = np.float32(np.random.rand(2, 100))
#  (W = [0.1, 0.2], b = 0.3)
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# b = 1 x 1 ,
b = tf.Variable(tf.zeros([1]))
# W= 1x2
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# loss function definition
loss = tf.reduce_mean(tf.square(y - y_data))

# make Gradient Descent (0.5 = learning rate)
optimizer = tf.train.GradientDescentOptimizer(0.5)
#
train = optimizer.minimize(loss)

#
init = tf.initialize_all_variables()

#
sess = tf.Session()
sess.run(init)

# 200 cycle.
for step in xrange(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(b)