from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

sys.path.append("../utils/")
import tf_utils


def polygon_encoder_decoder(x_image, input_res, encoding_length, output_vertex_count, weight_decay=None):
    """
    Builds the graph for a deep net for encoding and decoding polygons.

    Args:
      x_image: input variable
      input_res: an input tensor with the dimensions (N_examples, input_res, input_res, 1)
      encoding_length: number of neurons used in the bottleneck to encode the input polygon
      output_vertex_count: number of vertex of the polygon output
      weight_decay: Weight decay coefficient

    Returns:
      y: tensor of shape (N_examples, output_vertex_count, 2), with vertex coordinates
      keep_prob: scalar placeholder for the probability of dropout.
    """
    # with tf.name_scope('reshape'):
    #     x_image = tf.reshape(x, [-1, input_res, input_res, 1])

    # First convolutional layer - maps one grayscale image to 8 feature maps.
    with tf.name_scope('Features'):
        with tf.name_scope('conv1'):
            conv1 = tf_utils.complete_conv2d(x_image, 5, 1, 8, weight_decay)

        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            h_pool1 = tf_utils.max_pool_2x2(conv1)

        # Second convolutional layer -- maps 8 feature maps to 16.
        with tf.name_scope('conv2'):
            conv2 = tf_utils.complete_conv2d(h_pool1, 5, 8, 16, weight_decay)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            h_pool2 = tf_utils.max_pool_2x2(conv2)

        # Third convolutional layer -- maps 16 feature maps to 32.
        with tf.name_scope('conv3'):
            conv3 = tf_utils.complete_conv2d(h_pool2, 5, 16, 32, weight_decay)

        # Third pooling layer.
        with tf.name_scope('pool3'):
            h_pool3 = tf_utils.max_pool_2x2(conv3)

    current_shape = h_pool3.shape
    current_data_dimension = int(current_shape[1] * current_shape[2] * current_shape[3])

    with tf.name_scope('Encoder'):
        with tf.name_scope('flatten'):
            h_pool3_flat = tf.reshape(h_pool3, [-1, current_data_dimension])

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            # tf.summary.scalar('dropout_keep_probability', keep_prob)
            h_pool3_flat_drop = tf.nn.dropout(h_pool3_flat, keep_prob)

        with tf.name_scope('fc1'):
            fc1 = tf_utils.complete_fc(h_pool3_flat_drop, current_data_dimension, encoding_length, weight_decay, tf.nn.relu)

    y_coords = decode(fc1, encoding_length, output_vertex_count, weight_decay, scope_name="Decode")

    return y_coords, keep_prob


def decode(input_tensor, encoding_length, output_vertex_count, weight_decay, scope_name="Decode"):
    """
        Builds the graph for decoding polygons.

        Args:
          input_tensor: input variable (the encoding of a polygon)
          encoding_length: number of neurons used in the bottleneck to encode the input polygon
          output_vertex_count: number of vertex of the polygon output
          weight_decay: Weight decay coefficient
          scope_name: Scope under which this sub-graph will be built

        Returns:
          y_coords: tensor of shape (N_examples, output_vertex_count, 2), with vertex coordinates
        """
    with tf.name_scope(scope_name):
        with tf.name_scope('fc1'):
            fc1 = tf_utils.complete_fc(input_tensor, encoding_length, 256, weight_decay, tf.nn.relu)

        with tf.name_scope('fc2'):
            fc2 = tf_utils.complete_fc(fc1, 256, output_vertex_count * 2, weight_decay, tf.nn.sigmoid)

        with tf.name_scope('reshape_output'):
            y_coords = tf.reshape(fc2, [-1, output_vertex_count, 2])

    return y_coords
