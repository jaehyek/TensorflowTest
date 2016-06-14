'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
import random

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.02
training_epochs = 500
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # note : batch_ys is label  of which number 3  format is  [0,0,0,1,0,0,0,0,0,0 ]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy : ", accuracy.eval({x:mnist.test.images, y:mnist.test.labels})

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples -1 )
    print "Label : ", sess.run( tf.argmax(mnist.test.labels[r:r+1], 1))
    print "predicton : ", sess.run( tf.argmax(pred, 1 ), feed_dict={x:mnist.test.images[r:r+1]})

    countfail = 0
    # mnist.test.num_examples = 10000
    countloop = 2000
    for step in xrange( countloop) :
        label = sess.run( tf.argmax(mnist.test.labels[step:step+1], 1))
        output = sess.run( tf.argmax(pred, 1 ), feed_dict={x:mnist.test.images[step:step+1]})
        # print step, "/", countloop, " ",
        # print "Label : ", label,
        # print "predicton : ", output,
        countfail +=  0  if (label[0] == output[0]) else 1
        # print "match : %s" %( "pass" if (label[0] == output[0]) else "fail")

    print "training_epochs : ", training_epochs,
    print "total fail count : ", countfail, " / ", countloop,
    print "percentage : ",  float(countfail)/countloop
    # Test model
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

"""
Epoch: 0001 cost= 1.182138980, training_epochs :  1 total fail count :  364  /  2000 percentage :  0.182
Epoch: 0010 cost= 0.392387920, training_epochs :  10 total fail count :  256  /  2000 percentage :  0.128
Epoch: 0025 cost= 0.333687109, training_epochs :  25 total fail count :  226  /  2000 percentage :  0.113
Epoch: 0100 cost= 0.283246351, training_epochs :  100 total fail count :  208  /  2000 percentage :  0.104
Epoch: 0500 cost= 0.250752013, training_epochs :  500 total fail count :  198  /  2000 percentage :  0.099
Epoch: 0500 cost= 0.240970530, training_epochs :  500 total fail count :  195  /  2000 percentage :  0.0975  learning_rate = 0.02
"""