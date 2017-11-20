from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from collections import Counter
from model import get_convnet_tensors, add_adversarial_segment


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', default=1, type=int, help='Whether logging is verbose (1) or not (0).')
    parser.add_argument('--victim_class', default=2, type=int, help='Victim class of the adversarial attack.')
    parser.add_argument('--wanted_class', default=6, type=int, help='Class to misclassify the victim samples for.')
    parser.add_argument('--num_iterations', default=30, type=int,
        help='Maximum number of iterations used for optimizing adversarial examples.')
    parser.add_argument('--num_viz_samples', default=10, type=int, help='Number of samples to visualize.')
    parser.add_argument('--viz_path', default='adv_grid_viz.png', type=str, help='Path to save the visualization.')
    return parser.parse_args()

MODEL_PATH = 'data/mnist_conv/mnist_conv'
args = parse_args()

maybe_print = print if args.verbose else lambda *args, **kwargs: None

def pick_k(k, *arrays):
    import random
    choices = random.sample(population=range(arrays[0].shape[0]), k=k)
    return map(lambda a: a[choices], arrays)

def save_visualization(victim_x, modified_x):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    assert victim_x.shape == modified_x.shape
    num_x = victim_x.shape[0]
    fig = plt.figure(figsize=(3, num_x))
    gs = gridspec.GridSpec(nrows=num_x, ncols=3)

    victim_x, modified_x = victim_x.reshape((-1, 28, 28)), modified_x.reshape((-1, 28, 28))
    x_diff = modified_x - victim_x

    def plot(im, pos, label=None):
        im = np.clip(im, 0.0, 1.0)
        ax = fig.add_subplot(gs[pos])
        ax.imshow(im, cmap='gray', interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        if label:
            ax.set_xlabel(label, fontsize=10)

    for i in xrange(num_x):
        plot(victim_x[i], (i, 0,), label='Original' if i + 1 == num_x else None)
        plot(modified_x[i], (i, 1), label='Modified' if i + 1 == num_x else None)
        plot(x_diff[i], (i, 2), label='Difference' if i + 1 == num_x else None)

    gs.tight_layout(fig)
    plt.savefig(args.viz_path)
    maybe_print('Figure is saved at', args.viz_path)

maybe_print(
    'Will create adversarial samples of {} to be misclassified as {}'.format(args.victim_class, args.wanted_class))

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('{}.meta'.format(MODEL_PATH))
    saver.restore(sess, MODEL_PATH)
    x_, x, y, is_training, logits, pred, accuracy = get_convnet_tensors()
    adv_targeted, adv_eps, adv_x, adv_y = add_adversarial_segment(x, logits)

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    maybe_print('Test accuracy %g' % accuracy.eval(feed_dict={
        x_: mnist.test.images, y: mnist.test.labels, is_training: False}))
    test_pred = pred.eval(feed_dict={x_: mnist.test.images, is_training: False})

    current_class, wanted_class = args.victim_class, args.wanted_class

    # Only pick samples of the victim class that the net was able to classify correctly
    victim_mask = np.argmax(mnist.test.labels, axis=1) == current_class
    victim_mask &= test_pred == current_class
    victim_x, victim_y = mnist.test.images[victim_mask], mnist.test.labels[victim_mask]
    num_victims = victim_x.shape[0]
    maybe_print('Total num victim samples:', num_victims)

    modified_x = victim_x
    wanted_y = [wanted_class] * num_victims
    # NOTE: Keep in mind that successfully misclassified samples are no longer distorted
    for i in xrange(args.num_iterations):
        modified_x, predicted_classes = sess.run([adv_x, pred], feed_dict={
            x_: modified_x,
            adv_targeted: True,
            adv_eps: 0.01,
            adv_y: wanted_y,
            is_training: False
        })
        i += 1
        maybe_print('Iteration:', i, end=', ')
        maybe_print('Misclassification rate:', np.sum(predicted_classes != current_class) / float(num_victims), end=', ')
        maybe_print('Targeted misclassification rate:', np.sum(predicted_classes == wanted_class) / num_victims)

    # Only visualize the succesful adversarial examples
    success_mask = predicted_classes == wanted_class
    victim_x, modified_x = victim_x[success_mask], modified_x[success_mask]

    maybe_print('Creating and saving the visualizations for {} randomly chosen samples.'.format(args.num_viz_samples))
    save_visualization(*pick_k(args.num_viz_samples, victim_x, modified_x))
