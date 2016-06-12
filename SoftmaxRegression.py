__author__ = 'jaehyek'


import tensorflow as tf
import numpy as np

xy = np.loadtxt('train3.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

X = tf.placeholder(dtype="float32", shape=[None, 3])
Y = tf.placeholder(dtype="float32", shape=[None, 3])

W = tf.Variable(tf.zeros([3,3]))

print x_data
print y_data
print W

mul = tf.matmul(X, W)
hypothsis = tf.nn.softmax(tf.matmul(X, W))

# loss function definition
costlog = -tf.log(hypothsis)
costsum = tf.reduce_sum(Y * -1 * tf.log(hypothsis), reduction_indices=1 )
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothsis), reduction_indices=1 ))


# make Gradient Descent (0.5 = learning rate)
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 200 cycle.
for step in xrange(300001):
    sess.run(optimizer, feed_dict={X:x_data, Y:y_data})

    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}) # sess.run(W)

# print "-------------------------------------------------"
# a = sess.run(hypothsis, feed_dict={X:[[1,11,7]]})
# print a, sess.run(tf.arg_max(a, 1))
#
# b = sess.run(hypothsis, feed_dict={X:[[1,3,4]]})
# print b, sess.run(tf.arg_max(b, 1))
#
# c = sess.run(hypothsis, feed_dict={X:[[1,1,0]]})
# print c, sess.run(tf.arg_max(c, 1))

