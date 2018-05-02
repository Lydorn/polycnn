from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import matplotlib.pyplot as plt

import tensorflow as tf
import os

sys.path.append("../utils/")
import polygon_utils
import tf_utils

import model
import dataset
import loss

FLAGS = None

# --- Params --- #


ROOT_DIR = "../../"

# Data
input_res = 64
input_channels = 3
INPUT_DYNAMIC_RANGE = [-1, 1]  # [min, max] the network expects
output_vertex_count = 4
TFRECORDS_DIR = os.path.join(ROOT_DIR, "data/photovoltaic_array_location_dataset/tfrecords.unet_and_vectorization")

# Model

# Training
LEARNING_RATE_PARAMS = {
    "boundaries": [50000],
    "values": [1e-4, 1e-4]
}
batch_size = 128
max_iter = 50000
correct_dist_threshold = 1 / input_res  # 1px

# Validation
# The following value has to be equal to batch_size because the effective size of any batch has to be equal to
# batch_size (the custom loss function needs this condition)
dataset_val_size = 256

train_loss_accuracy_steps = 50
val_loss_accuracy_steps = 250
checkpoint_steps = 250

# Outputs
model_name = "unet-and-vectorization-photovoltaic-arrays"
LOGS_DIR = os.path.join(ROOT_DIR, "code/unet_and_vectorization/runs/current/logs")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "code/unet_and_vectorization/runs/current/checkpoints")


# --- --- #

def init_plots():
    fig1 = plt.figure(1, figsize=(5, 5))
    fig1.canvas.set_window_title('Training')
    fig2 = plt.figure(2, figsize=(5, 5))
    fig2.canvas.set_window_title('Validation')
    plt.ion()


def plot_results(figure_index, image_batch, polygon_batch, y_image_batch):
    im_res = image_batch[0].shape[0]
    image = (image_batch[0] - INPUT_DYNAMIC_RANGE[0]) / (INPUT_DYNAMIC_RANGE[1] - INPUT_DYNAMIC_RANGE[0])
    y_image = y_image_batch[0]
    train_polygon = polygon_batch[0] * im_res
    plt.figure(figure_index)
    plt.cla()
    plt.imshow(image)
    plt.imshow(y_image[:, :, 0], alpha=0.5, cmap="gray")
    polygon_utils.plot_polygon(train_polygon, label_direction=1)
    plt.draw()
    plt.pause(0.001)


def main(_):
    # Create the input placeholder
    x_image = tf.placeholder(tf.float32, [batch_size, input_res, input_res, input_channels])

    # Define loss and optimizer
    y_image_ = tf.placeholder(tf.float32, [batch_size, input_res, input_res, 1])

    y_image, mode_training = model.make_unet(x_image=x_image)

    # Build the objective loss function as well as the accuracy parts of the graph
    total_loss = loss.cross_entropy(y_image, y_image_)
    tf.summary.scalar('total_loss', total_loss)

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    learning_rate = tf.train.piecewise_constant(global_step, LEARNING_RATE_PARAMS["boundaries"],
                                                LEARNING_RATE_PARAMS["values"])

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

    # Summaries
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(LOGS_DIR, "train"), tf.get_default_graph())
    val_writer = tf.summary.FileWriter(os.path.join(LOGS_DIR, "val"), tf.get_default_graph())

    # Dataset
    train_dataset_filename = os.path.join(TFRECORDS_DIR, "train.tfrecord")
    train_images, train_polygons, train_raster_polygons = dataset.read_and_decode(train_dataset_filename, input_res,
                                                                                  output_vertex_count, batch_size,
                                                                                  INPUT_DYNAMIC_RANGE)
    val_dataset_filename = os.path.join(TFRECORDS_DIR, "val.tfrecord")
    val_images, val_polygons, val_raster_polygons = dataset.read_and_decode(val_dataset_filename, input_res,
                                                                            output_vertex_count, batch_size,
                                                                            INPUT_DYNAMIC_RANGE,
                                                                            augment_dataset=False)

    # Savers
    saver = tf.train.Saver()

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        # Restore checkpoint if one exists
        checkpoint = tf.train.get_checkpoint_state(CHECKPOINTS_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:  # First check if the whole model has a checkpoint
            print("Restoring {} checkpoint {}".format(model_name, checkpoint.model_checkpoint_path))
            saver.restore(sess, checkpoint.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        init_plots()

        print("Model has {} trainable variables".format(
            tf_utils.count_number_trainable_params())
        )

        i = tf.train.global_step(sess, global_step)
        while i <= max_iter:
            train_image_batch, train_polygon_batch, train_raster_polygon_batch = sess.run(
                [train_images, train_polygons, train_raster_polygons])
            if i % train_loss_accuracy_steps == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                train_summary, _, train_loss, train_y_image = sess.run(
                    [merged_summaries, train_step, total_loss, y_image],
                    feed_dict={x_image: train_image_batch, y_image_: train_raster_polygon_batch,
                               mode_training: True}, options=run_options, run_metadata=run_metadata)
                train_writer.add_summary(train_summary, i)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                print('step %d, training loss = %g' % (i, train_loss))
                plot_results(1, train_image_batch, train_polygon_batch, train_y_image)
            else:
                _ = sess.run([train_step], feed_dict={x_image: train_image_batch, y_image_: train_raster_polygon_batch,
                                                      mode_training: True})

            # Measure validation loss and accuracy
            if i % val_loss_accuracy_steps == 1:
                val_image_batch, val_polygon_batch, val_raster_polygon_batch = sess.run(
                    [val_images, val_polygons, val_raster_polygons])
                val_summary, val_loss, val_y_image = sess.run(
                    [merged_summaries, total_loss, y_image],
                    feed_dict={
                        x_image: val_image_batch,
                        y_image_: val_raster_polygon_batch, mode_training: True})
                val_writer.add_summary(val_summary, i)

                print('step %d, validation loss = %g' % (i, val_loss))
                plot_results(2, val_image_batch, val_polygon_batch, val_y_image)

            # Save checkpoint
            if i % checkpoint_steps == (checkpoint_steps - 1):
                saver.save(sess, os.path.join(CHECKPOINTS_DIR, model_name),
                           global_step=global_step)

            i = tf.train.global_step(sess, global_step)

        coord.request_stop()
        coord.join(threads)

        train_writer.close()
        val_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
