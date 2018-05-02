from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

sys.path.append("../utils/")
import tf_utils

sys.path.append("../../models/inception/")
import inception_v4

# --- Params --- #

# --- --- #


def feature_extractor_polygon_regressor(x_image, feature_extractor_name, encoding_length, output_vertex_count, weight_decay=None):

    features = feature_extractor(x_image, scope_name=feature_extractor_name)

    fc1, keep_prob = encode(features, encoding_length, weight_decay, scope_name="Encode")

    # Get Inception V4 variables used to restore checkpoint later
    # inceptionv4_variables = [var for var in tf.global_variables()]

    # --- Decoder --- #
    y_coords = decode(fc1, encoding_length, output_vertex_count, weight_decay, scope_name="Decode")

    # polygon_regressor_variables = [var for var in tf.global_variables() if var not in inceptionv4_variables]

    return y_coords, keep_prob


def feature_extractor(input_tensor, scope_name="FeatureExtractor"):
    last_layer_name = 'Mixed_5e'
    with tf.contrib.slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        inception_output, end_points = inception_v4.inception_v4_base(input_tensor, final_endpoint=last_layer_name,
                                                                      scope=scope_name)
    return inception_output


def encode(input_tensor, encoding_length, weight_decay, scope_name="Encode"):

    input_tensor_shape = input_tensor.shape
    input_tensor_channel_count = int(input_tensor_shape[3])

    with tf.name_scope(scope_name):
        with tf.name_scope('conv1'):
            conv1 = tf_utils.complete_conv2d(input_tensor, 1, input_tensor_channel_count, 512, weight_decay)

        with tf.name_scope('conv2'):
            conv2 = tf_utils.complete_conv2d(conv1, 1, 512, 128, weight_decay)

        current_output_shape = conv2.shape
        current_data_dimension = int(current_output_shape[1]*current_output_shape[2]*current_output_shape[3])

        with tf.name_scope('flatten'):
            conv2_flat = tf.reshape(conv2, [-1, current_data_dimension])

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            # tf.summary.scalar('dropout_keep_probability', keep_prob)
            inception_output_drop = tf.nn.dropout(conv2_flat, keep_prob)

        with tf.name_scope('fc1'):
            fc1 = tf_utils.complete_fc(inception_output_drop, current_data_dimension, encoding_length, weight_decay, tf.nn.relu)

    return fc1, keep_prob


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


def polygon_regressor(x_image, input_res, input_channels, encoding_length, output_vertex_count, weight_decay):
    """
    Builds the graph for a deep net for encoding and decoding polygons.

    Args:
      x_image: input tensor of shape (N_examples, input_res, input_res, input_channels)
      input_res: image resolution
      input_channels: image number of channels
      encoding_length: number of neurons used in the bottleneck to encode the input polygon
      output_vertex_count: number of vertex of the polygon output
      weight_decay: Weight decay coefficient

    Returns:
      y: tensor of shape (N_examples, output_vertex_count, 2), with vertex coordinates
      keep_prob: scalar placeholder for the probability of dropout.
    """
    # with tf.name_scope('reshape'):
    #     x_image = tf.reshape(x, [-1, input_res, input_res, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        conv1 = tf_utils.complete_conv2d(x_image, 3, input_channels, 16, weight_decay)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = tf_utils.max_pool_2x2(conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        conv2 = tf_utils.complete_conv2d(h_pool1, 3, 16, 32, weight_decay)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = tf_utils.max_pool_2x2(conv2)

    # Third convolutional layer -- maps 64 feature maps to 128.
    with tf.name_scope('conv3'):
        conv3 = tf_utils.complete_conv2d(h_pool2, 3, 32, 64, weight_decay)

    # Second pooling layer.
    with tf.name_scope('pool3'):
        h_pool3 = tf_utils.max_pool_2x2(conv3)

    reduction_factor = 8  # Adjust according to previous layers
    current_data_dimension = int(input_res / reduction_factor) * int(input_res / reduction_factor) * 64

    with tf.name_scope('flatten'):
        h_pool3_flat = tf.reshape(h_pool3, [-1, current_data_dimension])

    # Fully connected layer 1 -- after 2 round of downsampling, our 64x64 image
    # is down to 8x8x128 feature maps -- map this to 2048 features.
    with tf.name_scope('fc1'):
        fc1 = tf_utils.complete_fc(h_pool3_flat, current_data_dimension, 1024, weight_decay, tf.nn.relu)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        # tf.summary.scalar('dropout_keep_probability', keep_prob)
        fc1_drop = tf.nn.dropout(fc1, keep_prob)

    # Map the 2048 features to encoding_length features
    with tf.name_scope('fc2'):
        fc2 = tf_utils.complete_fc(fc1_drop, 1024, encoding_length, weight_decay, tf.nn.relu)

    # --- Decoder --- #

    # Map the encoding_length features to 2048 features
    with tf.name_scope('fc3'):
        fc3 = tf_utils.complete_fc(fc2, encoding_length, 512, weight_decay, tf.nn.relu)

    # Map the 2048 features to the output_vertex_count * 2 output coordinates
    with tf.name_scope('fc4'):
        y_flat = tf_utils.complete_fc(fc3, 512, output_vertex_count * 2, weight_decay, tf.nn.sigmoid)

    with tf.name_scope('reshape_output'):
        y_coords = tf.reshape(y_flat, [-1, output_vertex_count, 2])

    return y_coords, keep_prob


