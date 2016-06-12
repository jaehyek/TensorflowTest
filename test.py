__author__ = 'jaehyek'
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
costsum = tf.reduce_sum(X,reduction_indices=1)
cost = tf.reduce_mean(X)


# make Gradient Descent (0.5 = learning rate)
# learning_rate = 0.01
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print "costsum \n", sess.run(costsum, feed_dict={X:x_data, Y:y_data})
print "cost \n", sess.run(cost, feed_dict={X:x_data, Y:y_data})

exit()

# 200 cycle.
for step in xrange(5):
    # print "W \n", sess.run(W)
    # print "mul \n", sess.run(mul, feed_dict={X:x_data, Y:y_data})
    # print "hyposthsis \n", sess.run(hypothsis, feed_dict={X:x_data, Y:y_data})
    # print "costlog \n", sess.run(costlog, feed_dict={X:x_data, Y:y_data})
    print "costsum \n", sess.run(costsum, feed_dict={X:x_data, Y:y_data})
    print "cost \n", sess.run(cost, feed_dict={X:x_data, Y:y_data})


    sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
    print "-------------------------------"


    # if step % 20 == 0:
    #     print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)

