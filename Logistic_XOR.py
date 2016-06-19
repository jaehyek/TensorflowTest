__author__ = 'jaehyek'

import tensorflow as tf
import numpy as np

xy = np.loadtxt('train_nor.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:-2])
y_data = np.transpose(xy[-2:-1])

print x_data
print y_data


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2,10], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([10,1], -1.0, 1.0))
B1 = tf.Variable(tf.zeros( [10] ) )
B2 = tf.Variable(tf.zeros( [1] ) )

L2 = tf.sigmoid(tf.matmul(X, W1) + B1)
hypothsis = tf.sigmoid(tf.matmul(L2, W2) + B2)
# loss function definition
cost = -tf.reduce_mean(Y * tf.log(hypothsis) + (1-Y)*tf.log(1-hypothsis))

# make Gradient Descent (0.5 = learning rate)
a = tf.Variable(0.1)
train = tf.train.GradientDescentOptimizer(a).minimize(cost)

init = tf.initialize_all_variables()

#
with tf.Session() as sess :
    sess.run(init)

    # 200 cycle.
    for step in xrange(10000):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1),sess.run(B1),sess.run(W2),sess.run(B2)
            print "------------------------------------------"

    # test model
    correct_prediction = tf.equal(tf.floor(hypothsis+0.5), Y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "------------------hypothsis------------------------"
    print sess.run([hypothsis],feed_dict={X:x_data, Y:y_data} )
    print "------------------floor------------------------"
    print sess.run([tf.floor(hypothsis+0.5)],feed_dict={X:x_data, Y:y_data} )
    print "------------------correct_prediction------------------------"
    print sess.run([ correct_prediction],feed_dict={X:x_data, Y:y_data} )
    print "------------------Accuracy------------------------"
    print "Accuracy : ", accuracy.eval({X:x_data, Y:y_data})