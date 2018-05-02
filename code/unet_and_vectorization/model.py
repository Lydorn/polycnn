"""
Simple U-Net implementation in TensorFlow initially by Mo Kweon (https://github.com/kkweon/UNet-in-Tensorflow)

Original Paper:
    https://arxiv.org/abs/1505.04597
"""
import time
import os
import tensorflow as tf


def conv_conv_pool(input_, n_filters, training, name, pool=True, activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}

    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions

    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(net, F, (3, 3), activation=None, padding='same', name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


def upsample_concat(inputA, input_B, name):
    """Upsample `inputA` and concat with `input_B`

    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation

    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    upsample = upsampling_2D(inputA, size=(2, 2), name=name)

    return tf.concat([upsample, input_B], axis=-1, name="concat_{}".format(name))


def upsampling_2D(tensor, name, size=(2, 2)):
    """Upsample/Rescale `tensor` by size

    Args:
        tensor (4-D Tensor): (N, H, W, C)
        name (str): name of upsampling operations
        size (tuple, optional): (height_multiplier, width_multiplier)
            (default: (2, 2))

    Returns:
        output (4-D Tensor): (N, h_multiplier * H, w_multiplier * W, C)
    """
    H, W, _ = tensor.get_shape().as_list()[1:]

    H_multi, W_multi = size
    target_H = H * H_multi
    target_W = W * W_multi

    return tf.image.resize_nearest_neighbor(tensor, (target_H, target_W), name="upsample_{}".format(name))


def make_unet(x_image):
    """Build a U-Net architecture

    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers

    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor

    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    training = tf.placeholder(tf.bool, name="training")

    feature_base_count = 4

    color_space_adjust = tf.layers.conv2d(x_image, 3, (1, 1), name="color_space_adjust")
    conv1, pool1 = conv_conv_pool(color_space_adjust, [feature_base_count, feature_base_count], training, name="1")
    conv2, pool2 = conv_conv_pool(pool1, [feature_base_count*2, feature_base_count*2], training, name="2")
    conv3, pool3 = conv_conv_pool(pool2, [feature_base_count*4, feature_base_count*4], training, name="3")
    conv4, pool4 = conv_conv_pool(pool3, [feature_base_count*8, feature_base_count*8], training, name="4")
    conv5 = conv_conv_pool(pool4, [feature_base_count*16, feature_base_count*16], training, name="5", pool=False)

    up6 = upsample_concat(conv5, conv4, name="6")
    conv6 = conv_conv_pool(up6, [feature_base_count*8, feature_base_count*8], training, name="6", pool=False)

    up7 = upsample_concat(conv6, conv3, name="7")
    conv7 = conv_conv_pool(up7, [feature_base_count*4, feature_base_count*4], training, name="7", pool=False)

    up8 = upsample_concat(conv7, conv2, name="8")
    conv8 = conv_conv_pool(up8, [feature_base_count*2, feature_base_count*2], training, name="8", pool=False)

    up9 = upsample_concat(conv8, conv1, name="9")
    conv9 = conv_conv_pool(up9, [feature_base_count, feature_base_count], training, name="9", pool=False)

    # output = tf.layers.conv2d(conv9, 1, (1, 1), name='final', activation=tf.nn.sigmoid, padding='same')
    output = tf.layers.conv2d(conv9, 1, (1, 1), name='final', activation=tf.identity, padding='same')

    return output, training
