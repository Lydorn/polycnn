import tensorflow as tf


def count_number_trainable_params(trainable_variables=None):
    """
    Counts the number of trainable variables.
    """
    if trainable_variables is None:
        trainable_variables = tf.trainable_variables()
    tot_nb_params = 0
    for trainable_variable in trainable_variables:
        shape = trainable_variable.get_shape()  # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params


def get_nb_params_shape(shape):
    """
    Computes the total number of params for a given shap.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    """
    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params


def complete_conv2d(input, kernel_size, input_channels, output_channels, weight_decay):
    with tf.name_scope('W'):
        w_conv = weight_variable([kernel_size, kernel_size, input_channels, output_channels], weight_decay)
        variable_summaries(w_conv)
    with tf.name_scope('bias'):
        b_conv = bias_variable([output_channels])
        variable_summaries(b_conv)
    z_conv = conv2d(input, w_conv) + b_conv
    tf.summary.histogram('pre_activations', z_conv)
    h_conv = tf.nn.relu(z_conv)
    tf.summary.histogram('activations', h_conv)
    return h_conv


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def complete_fc(input, input_channels, output_channels, weight_decay, activation_func):
    with tf.name_scope('W'):
        w_fc = weight_variable([input_channels, output_channels], weight_decay)
        variable_summaries(w_fc)
    with tf.name_scope('bias'):
        b_fc = bias_variable([output_channels])
        variable_summaries(b_fc)
    z_fc = tf.matmul(input, w_fc) + b_fc
    tf.summary.histogram('pre_activations', z_fc)
    h_fc = activation_func(z_fc)
    tf.summary.histogram('activations', h_fc)
    return h_fc


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape, wd):
    """weight_variable generates a weight variable of a given shape. Adds weight decay if specified"""
    initial = tf.truncated_normal(shape, stddev=0.02)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(initial), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.02, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    # with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    # with tf.name_scope('stddev'):
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
