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

W1 = tf.Variable(tf.random_uniform([2,5], -1.0, 1.0), name="weight1")
W2 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="weight2")
W3 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="weight3")
W4 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="weight4")
W5 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="weight5")
W6 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="weight6")
W7 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="weight7")
W8 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="weight8")
W9 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="weight9")
W10 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="weight10")
W11 = tf.Variable(tf.random_uniform([5,1], -1.0, 1.0), name="weight11")


B1 = tf.Variable(tf.zeros( [5] ), name="bias1" )
B2 = tf.Variable(tf.zeros( [5] ), name="bias2" )
B3 = tf.Variable(tf.zeros( [5] ), name="bias3" )
B4 = tf.Variable(tf.zeros( [5] ), name="bias4" )
B5 = tf.Variable(tf.zeros( [5] ), name="bias5" )
B6 = tf.Variable(tf.zeros( [5] ), name="bias6" )
B7 = tf.Variable(tf.zeros( [5] ), name="bias7" )
B8 = tf.Variable(tf.zeros( [5] ), name="bias8" )
B9 = tf.Variable(tf.zeros( [5] ), name="bias9" )
B10 = tf.Variable(tf.zeros( [5] ), name="bias10" )
B11 = tf.Variable(tf.zeros( [1] ), name="bias11" )

with tf.name_scope("layer1") as scope :
    L1 = tf.nn.relu(tf.matmul(X, W1) + B1)
    tf.histogram_summary("layer1", L1)
with tf.name_scope("layer2") as scope :
    L2 = tf.nn.relu(tf.matmul(L1, W2) + B2)
    tf.histogram_summary("layer2", L2)
with tf.name_scope("layer3") as scope :
    L3 = tf.nn.relu(tf.matmul(L2, W3) + B3)
    tf.histogram_summary("layer3", L3)
with tf.name_scope("layer4") as scope :
    L4 = tf.nn.relu(tf.matmul(L3, W4) + B4)
    tf.histogram_summary("layer4", L4)
with tf.name_scope("layer5") as scope :
    L5 = tf.nn.relu(tf.matmul(L4, W5) + B5)
    tf.histogram_summary("layer5", L5)
with tf.name_scope("layer6") as scope :
    L6 = tf.nn.relu(tf.matmul(L5, W6) + B6)
    tf.histogram_summary("layer6", L6)
with tf.name_scope("layer7") as scope :
    L7 = tf.nn.relu(tf.matmul(L6, W7) + B7)
    tf.histogram_summary("layer7", L7)
with tf.name_scope("layer8") as scope :
    L8 = tf.nn.relu(tf.matmul(L7, W8) + B8)
    tf.histogram_summary("layer8", L8)
with tf.name_scope("layer9") as scope :
    L9 = tf.nn.relu(tf.matmul(L8, W9) + B9)
    tf.histogram_summary("layer9", L9)
with tf.name_scope("layer10") as scope :
    L10 = tf.nn.relu(tf.matmul(L9, W10) + B10)
    tf.histogram_summary("layer10", L10)
with tf.name_scope("last") as scope :
    hypothsis = tf.sigmoid(tf.matmul(L10, W11) + B11)
    tf.histogram_summary("last", hypothsis)


# loss function definition
cost = -tf.reduce_mean(Y * tf.log(hypothsis) + (1-Y)*tf.log(1-hypothsis))
tf.scalar_summary("cost", cost)
# make Gradient Descent (0.5 = learning rate)
a = tf.Variable(0.1)
train = tf.train.GradientDescentOptimizer(a).minimize(cost)


init = tf.initialize_all_variables()

#
with tf.Session() as sess :
    sess.run(init)
    merged = tf.merge_all_summaries()
    print merged
    writer = tf.train.SummaryWriter("./logs", sess.graph)

    # 200 cycle.
    for step in xrange(10000):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data} )
            summary = sess.run(merged, feed_dict={X:x_data, Y:y_data} )
            writer.add_summary(summary, step)
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
