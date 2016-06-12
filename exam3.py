__author__ = 'jaehyek'

import tensorflow as tf
import numpy as np

xy = np.loadtxt('train2.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

print x_data
print y_data
W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

Hypothsis = tf.matmul(W, x_data)
# loss function definition
cost = tf.reduce_mean(tf.square(Hypothsis - y_data))

# make Gradient Descent (0.5 = learning rate)
a = tf.Variable(0.1)
train = tf.train.GradientDescentOptimizer(a).minimize(cost)

init = tf.initialize_all_variables()

#
sess = tf.Session()
sess.run(init)

# 200 cycle.
for step in xrange(2001):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(W)
