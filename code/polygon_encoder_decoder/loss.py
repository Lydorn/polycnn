from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf

FLAGS = None


def generate_expand_matrix(batch_size, n):
    mat_identity = np.eye(n, dtype=np.float32)
    mat_expand = np.tile(mat_identity, (batch_size, n, 1))
    return mat_expand


def generate_rolls_matrix(batch_size, n):
    mat_identity = np.eye(n, dtype=np.float32)
    mat_rolls = mat_identity
    for i in range(1, n):
        mat_roll = np.roll(mat_identity, i, axis=0)
        mat_rolls = np.append(mat_rolls, mat_roll, axis=0)
    mat_rolls = np.tile(mat_rolls, (batch_size, 1, 1))
    return mat_rolls


def polygon_l2diffs(y_, y, batch_size):
    """
    The loss is the minimum of the MSE of all possible shifts between the groundtruth polygon and the predicted
    polygon. This loss is made to not penalize the model for predicting a correct polygon whose starting vertex is
    not the same as the groundtruth's

    Keyword arguments:
    y_ -- Groundtruth polygon (is actually interchangeable with y)
    y  -- predicted polygon (is actually interchangeable with y_)
    """
    assert y_.shape[1] == y.shape[1]
    assert y_.shape[2] == y.shape[2]
    assert y_.shape[2] == 2  # 2D polygon
    vertex_count = int(y_.shape[1])

    mat_expand = generate_expand_matrix(batch_size, vertex_count)
    mat_rolls = generate_rolls_matrix(batch_size, vertex_count)

    l2diffs_expended = tf.sqrt(
        tf.reduce_sum(
            tf.square(
                tf.subtract(tf.matmul(mat_expand, y_,), tf.matmul(mat_rolls, y))
            ), axis=2)
    )
    l2diffs = tf.reshape(l2diffs_expended, [batch_size, vertex_count, vertex_count])

    return l2diffs


def loss_and_accuracy(y_, y, batch_size, correct_dist_threshold):
    with tf.name_scope('loss'):
        l2diffs = polygon_l2diffs(y_, y, batch_size)
        mses = tf.reduce_mean(l2diffs, axis=2)
        losses = tf.reduce_min(mses, axis=1)
        objective_loss = tf.reduce_mean(losses)
        tf.summary.scalar('objective_loss', objective_loss)

    with tf.name_scope('accuracy'):
        min_index = tf.argmin(mses, axis=1)
        min_index = tf.reshape(min_index, [batch_size, 1])
        aligned_poly_l2diffs = tf.map_fn(lambda x: tf.gather(x[0], x[1]), (l2diffs, min_index), dtype=tf.float32)
        correct_prediction = tf.less(aligned_poly_l2diffs, correct_dist_threshold)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        tf.summary.scalar('accuracy', accuracy)

    return objective_loss, accuracy


def main(_):
    batch_size = 2
    correct_dist_threshold = 1e-6
    vertex_count = 4
    y_ = tf.placeholder(tf.float32, [batch_size, vertex_count, 2])
    y = tf.placeholder(tf.float32, [batch_size, vertex_count, 2])

    loss, accuracy = loss_and_accuracy(y_, y, batch_size, correct_dist_threshold)

    with tf.Session() as sess:
        gt = np.array([
            [[0, 0], [1, 1], [2, 2], [3, 3]],
            [[0, 0], [1, 1], [2, 2], [3, 3]]
        ])
        pred = np.array([
            [[1, 1], [1, 1], [2, 2], [3, 3]],
            [[1, 1], [2, 2], [3, 3], [0, 0]]
        ])
        # computed_loss = sess.run([loss], feed_dict={y_: gt, y: pred})
        # print(computed_loss)

        computed_loss, computed_accuracy = sess.run([loss, accuracy], feed_dict={y_: gt, y: pred})
        print(computed_loss)
        print(computed_accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
