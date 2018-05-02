from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
import os
# import matplotlib.pyplot as plt

from model import polygon_encoder_decoder
import dataset
import loss

sys.path.append("../utils/")
import polygon_utils
import tf_utils

FLAGS = None

# --- Params --- #
ROOT_DIR = "../../"

input_res = 64
INPUT_DYNAMIC_RANGE = [-1, 1]  # [min, max] the network expects
encoding_length = 128
output_vertex_count = 4
DATASET_DIR = os.path.join(ROOT_DIR, "data/polygon_encoder_decoder")

learning_rate = 1e-4
batch_size = 64
weight_decay = 1e-5
dropout_keep_prob = 1.0
max_iter = 20000
correct_dist_threshold = 1 / input_res  # 1px

# The following value has to be equal to batch_size because the effective size of any batch has to be equal to
# batch_size (the custom loss function needs this condition)
dataset_val_size = 64

train_loss_accuracy_steps = 500
val_loss_accuracy_steps = 2500
checkpoint_steps = 2500

MODEL_NAME = "polygon-encoder-decoder"
LOGS_DIR = os.path.join(ROOT_DIR, "code/polygon_encoder_decoder/runs/current/logs")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "code/polygon_encoder_decoder/runs/current/checkpoints")

# --- --- #

#
# def init_plots():
#     fig1 = plt.figure(1, figsize=(5, 5))
#     fig1.canvas.set_window_title('Training')
#     fig2 = plt.figure(2, figsize=(5, 5))
#     fig2.canvas.set_window_title('Validation')
#     plt.ion()
#
#
# def plot_results(figure_index, train_image_batch, train_polygon_batch, train_y_coords_batch):
#     im_res = train_image_batch[0].shape[0]
#     im_channel_count = train_image_batch[0].shape[2]
#     image = (train_image_batch[0] - INPUT_DYNAMIC_RANGE[0]) / (INPUT_DYNAMIC_RANGE[1] - INPUT_DYNAMIC_RANGE[0])
#     train_polygon = train_polygon_batch[0] * im_res
#     train_y_coords = train_y_coords_batch[0] * im_res
#     plt.figure(figure_index)
#     plt.cla()
#     if im_channel_count == 1:
#         plt.imshow(image[:, :, 0])
#     else:
#         plt.imshow(image)
#     polygon_utils.plot_polygon(train_polygon, label_direction=1)
#     polygon_utils.plot_polygon(train_y_coords, label_direction=-1)
#     plt.draw()
#     plt.pause(0.001)


def main(_):
    # Create the model
    x_image = tf.placeholder(tf.float32, [batch_size, input_res, input_res, 1])

    # Define loss and optimizer
    y_coords_ = tf.placeholder(tf.float32, [batch_size, output_vertex_count, 2])

    # Build the graph for the deep net
    y_coords, keep_prob = polygon_encoder_decoder(x_image=x_image, input_res=input_res, encoding_length=encoding_length,
                                                  output_vertex_count=output_vertex_count, weight_decay=weight_decay)

    # Build the objective loss function as well as the accuracy parts of the graph
    objective_loss, accuracy = loss.loss_and_accuracy(y_coords_, y_coords, batch_size, correct_dist_threshold)
    tf.add_to_collection('losses', objective_loss)

    # Add all losses (objective loss + weigh loss for now)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    tf.summary.scalar('total_loss', total_loss)

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

    # Summaries
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(LOGS_DIR, "train"), tf.get_default_graph())
    val_writer = tf.summary.FileWriter(os.path.join(LOGS_DIR, "val"), tf.get_default_graph())

    # Dataset
    train_dataset_filename = os.path.join(DATASET_DIR, "train.tfrecord")
    train_images, train_polygons = dataset.read_and_decode(train_dataset_filename, input_res,
                                                           output_vertex_count, batch_size, INPUT_DYNAMIC_RANGE)
    val_dataset_filename = os.path.join(DATASET_DIR, "val.tfrecord")
    val_images, val_polygons = dataset.read_and_decode(val_dataset_filename, input_res,
                                                       output_vertex_count, batch_size, INPUT_DYNAMIC_RANGE)

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
            print("Restoring {} checkpoint {}".format(MODEL_NAME, checkpoint.model_checkpoint_path))
            saver.restore(sess, checkpoint.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # init_plots()

        print("Model has {} trainable variables".format(
            tf_utils.count_number_trainable_params())
        )

        i = tf.train.global_step(sess, global_step)
        while i <= max_iter:
            train_image_batch, train_polygon_batch = sess.run([train_images, train_polygons])
            if i % train_loss_accuracy_steps == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                train_summary, _, train_loss, train_accuracy, train_y_coords = sess.run(
                    [merged_summaries, train_step, total_loss, accuracy, y_coords],
                    feed_dict={x_image: train_image_batch, y_coords_: train_polygon_batch,
                               keep_prob: dropout_keep_prob}, options=run_options, run_metadata=run_metadata)
                train_writer.add_summary(train_summary, i)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                print('step %d, training loss = %g, accuracy = %g' % (i, train_loss, train_accuracy))
                # plot_results(1, train_image_batch, train_polygon_batch, train_y_coords)
            else:
                _ = sess.run([train_step], feed_dict={x_image: train_image_batch, y_coords_: train_polygon_batch,
                                                      keep_prob: dropout_keep_prob})

            # Measure validation loss and accuracy
            if i % val_loss_accuracy_steps == 1:
                val_image_batch, val_polygon_batch = sess.run([val_images, val_polygons])
                val_summary, val_loss, val_accuracy, val_y_coords = sess.run(
                    [merged_summaries, total_loss, accuracy, y_coords],
                    feed_dict={
                        x_image: val_image_batch,
                        y_coords_: val_polygon_batch, keep_prob: 1.0})
                val_writer.add_summary(val_summary, i)

                print('step %d, validation loss = %g, accuracy = %g' % (i, val_loss, val_accuracy))
                # plot_results(2, val_image_batch, val_polygon_batch, val_y_coords)

            # Save checkpoint
            if i % checkpoint_steps == (checkpoint_steps - 1):
                saver.save(sess, os.path.join(CHECKPOINTS_DIR, MODEL_NAME),
                           global_step=global_step)

            i = tf.train.global_step(sess, global_step)

        coord.request_stop()
        coord.join(threads)

        train_writer.close()
        val_writer.close()


if __name__ == '__main__':
    tf.app.run(main=main)
