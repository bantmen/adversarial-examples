from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def construct_convnet(num_classes=10, input_shape=(28, 28, 1)):
    """
    Simple convnet that should have ~1.5 error rate on mnist test set.
    """
    input_size_x, input_size_y, input_size_c = input_shape

    # Graph inputs
    x_ = tf.placeholder(tf.float32, shape=[None, input_size_x * input_size_y * input_size_c], name='raw_input')
    x = tf.reshape(x_, shape=[-1, input_size_x, input_size_y, input_size_c], name='input')
    y = tf.placeholder(tf.int32, shape=[None, num_classes], name='target_onehot_class')
    is_training = tf.placeholder(tf.bool, name='is_training')

    logits = _logits_from_x(x, is_training=is_training, num_classes=num_classes)

    # TODO: Effects of label smoothing on adversarial examples?
    cost = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits),
        name='cross_entropy')
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

    pred = tf.argmax(logits, 1, name='prediction')
    correct_prediction = tf.equal(pred, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    return x_, x, y, is_training, train_step, accuracy

def _logits_from_x(x, is_training, num_classes, keep_prob=0.5):
    net = slim.conv2d(x, 32, 3, scope='conv3_1')
    net = slim.max_pool2d(net, 2, scope='pool1')
    net = slim.conv2d(net, 32, 3, scope='conv3_2')
    net = slim.max_pool2d(net, 2, scope='pool2')
    net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training, scope='dropout1')
    net = slim.flatten(net)
    net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc1')
    net = tf.identity(net, name='logits')
    return net

def get_convnet_tensors():
    x_ = tf.get_default_graph().get_tensor_by_name('raw_input:0')
    x = tf.get_default_graph().get_tensor_by_name('input:0')
    y = tf.get_default_graph().get_tensor_by_name('target_onehot_class:0')
    is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')
    logits = tf.get_default_graph().get_tensor_by_name('logits:0')
    pred = tf.get_default_graph().get_tensor_by_name('prediction:0')
    accuracy = tf.get_default_graph().get_tensor_by_name('accuracy:0')
    return x_, x, y, is_training, logits, pred, accuracy

def add_adversarial_segment(x, logits):
    """
    Modified version of fast gradient sign method (fgsm) (https://arxiv.org/abs/1412.6572) to
    produce adversarial inputs from the given batch of inputs x.
    If a specific class is not targeted, i.e. if adversarial_targeted is false, then this method
    is equivalent to fgsm.
    Otherwise, fgsm is employed while the sign of perturbation is flipped.
    """
    # Graph inputs
    adv_targeted = tf.placeholder(tf.bool, shape=(), name='adversarial_targeted')
    adv_eps = tf.placeholder(tf.float32, shape=(), name='adversarial_eps')
    adv_y = tf.placeholder(tf.int32, shape=[None], name='adversarial_y')

    adv_ce = tf.losses.softmax_cross_entropy(
        onehot_labels=tf.one_hot(adv_y, depth=logits.shape[-1]),
        logits=logits, scope='adversarial_cross_entropy')
    adv_ce_grad = tf.gradients(ys=adv_ce, xs=x)[0]
    # Fix the gradient sign depending on whether an arbitrary or a specific class is targeted
    adv_ce_grad = tf.where(adv_targeted, -1 * adv_ce_grad, adv_ce_grad)
    normalized_grad = tf.sign(adv_ce_grad)
    adv_perturbation = adv_eps * normalized_grad

    # Don't distort the samples that are already misclassified as intended
    already_adversarial = tf.equal(tf.argmax(logits, 1), tf.cast(adv_y, tf.int64))
    already_adversarial = tf.cast(tf.logical_not(already_adversarial), tf.float32)
    adv_perturbation_mask = tf.tile(tf.reshape(already_adversarial, [-1, 1, 1, 1]), [1, 28, 28, 1])
    adv_perturbation = adv_perturbation * adv_perturbation_mask

    adv_x = tf.clip_by_value(x + adv_perturbation, 0.0, 1.0)
    adv_x = tf.identity(slim.flatten(adv_x), name='adversarial_x')

    return adv_targeted, adv_eps, adv_x, adv_y
