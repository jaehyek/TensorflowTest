__author__ = 'jaehyek'

import tensorflow as tf

x_data = [ 1., 2., 3.]
y_data = [ 1., 2., 3.]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
tf.histogram_summary("weights", W)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))
tf.scalar_summary("cost", cost)

descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W,X) - Y), X )))
update = W.assign(descent)

init = tf.initialize_all_variables()


with tf.Session() as sess :
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/root/log/mnist_logs", sess.graph )
    sess.run(init)

    for step in xrange(10):
        # print sess.run(descent, feed_dict={X:x_data, Y:y_data}),
        # print sess.run(W),
        # sess.run(cost, feed_dict={X:x_data, Y:y_data})
        sess.run(update, feed_dict={X:x_data, Y:y_data})
        summary = sess.run(merged, feed_dict={X:x_data, Y:y_data})

        # print sess.run(cost, feed_dict={X:x_data, Y:y_data})
        writer.add_summary(summary, step)