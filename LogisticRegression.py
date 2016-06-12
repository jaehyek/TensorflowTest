__author__ = 'jaehyek'

import tensorflow as tf
import numpy as np

xy = np.loadtxt('train2.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

print x_data
print y_data
print W

h = tf.matmul(W, x_data)
hypothsis = tf.div(1., 1.+tf.exp(-h))
# loss function definition
cost = -tf.reduce_mean(Y * tf.log(hypothsis) + (1-Y)*tf.log(1-hypothsis))


# make Gradient Descent (0.5 = learning rate)
a = tf.Variable(0.1)
train = tf.train.GradientDescentOptimizer(a).minimize(cost)

init = tf.initialize_all_variables()

#
sess = tf.Session()
sess.run(init)

# 200 cycle.
for step in xrange(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)