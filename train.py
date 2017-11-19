from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import subprocess
import os
import model


NUM_BATCHES = 20000
BATCH_SIZE = 50
MODEL_NAME = 'mnist_conv'
MODEL_DIR = 'data/{}'.format(MODEL_NAME)

# Make sure that model directory already exists
subprocess.call(['mkdir', '-p', MODEL_DIR])

x, _, y, is_training, train_step, accuracy = model.construct_convnet()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.Session() as sess:
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    for i in xrange(NUM_BATCHES):
        xbatch, y_batch = mnist.train.next_batch(BATCH_SIZE)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: xbatch, y: y_batch, is_training: False})
            print('Step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: xbatch, y: y_batch, is_training: True})
    print('Test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y: mnist.test.labels, is_training: False}))
    saver.save(sess, save_path=os.path.join(MODEL_DIR, MODEL_NAME), write_meta_graph=True)
