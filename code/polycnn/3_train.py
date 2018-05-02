from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
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

import matplotlib.pyplot as plt
ROOT_DIR = "../../"

# Data
input_res = 67
input_channels = 3
INPUT_DYNAMIC_RANGE = [-1, 1]  # [min, max] the network expects
output_vertex_count = 4
TFRECORDS_DIR = os.path.join(ROOT_DIR, "data/photovoltaic_array_location_dataset/tfrecords.polycnn")

# Model
ENCODING_LENGTH = 128
FEATURE_EXTRACTOR_PARAMS = {
    "name": "InceptionV4",
    "encoding_length": ENCODING_LENGTH,
    "use_pretrained_weights": True,  # Set to false to train from scratch (Default to True)
    "checkpoint_filepath": os.path.join(ROOT_DIR, "models/inception/inception_v4.ckpt")
}
DECODER_PARAMS = {
    "name": "Decode",
    "encoding_length": ENCODING_LENGTH,
    "use_pretrained_weights": True,  # Set to false to train from scratch (Default to True)
    "checkpoint_dir": os.path.join(ROOT_DIR, "code/polygon_encoder_decoder/runs/current/checkpoints")
}

# Training
LEARNING_RATE_PARAMS = {
    "boundaries": [100000],
    "values": [1e-5, 1e-6]
}
FEATURE_EXTRACTOR_LEARNING_RATE_PARAMS = {
    "boundaries": [1000, 100000],
    "values": [0.0, 1e-5, 1e-6]
}
DECODER_LEARNING_RATE_PARAMS = {
    "boundaries": [500, 100000],
    "values": [0.0, 1e-5, 1e-6]
}
batch_size = 128  # 256 was used for the result of the paper, but in retrospect it is a very large batch size
weight_decay = 1e-6  # Default: 1e-6
dropout_keep_prob = 1.0
max_iter = 200000
correct_dist_threshold = 1 / input_res  # 1px

train_loss_accuracy_steps = 50
val_loss_accuracy_steps = 250
checkpoint_steps = 250

# Outputs
model_name = "polycnn"
LOGS_DIR = os.path.join(ROOT_DIR, "code/polycnn/runs/current/logs")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "code/polycnn/runs/current/checkpoints")


# --- --- #

def init_plots():
    fig1 = plt.figure(1, figsize=(5, 5))
    fig1.canvas.set_window_title('Training')
    fig2 = plt.figure(2, figsize=(5, 5))
    fig2.canvas.set_window_title('Validation')
    plt.ion()


def plot_results(figure_index, train_image_batch, train_polygon_batch, train_y_coords_batch):
    im_res = train_image_batch[0].shape[0]
    image = (train_image_batch[0] - INPUT_DYNAMIC_RANGE[0]) / (INPUT_DYNAMIC_RANGE[1] - INPUT_DYNAMIC_RANGE[0])
    train_polygon = train_polygon_batch[0] * im_res
    train_y_coords = train_y_coords_batch[0] * im_res
    plt.figure(figure_index)
    plt.cla()
    plt.imshow(image)
    polygon_utils.plot_polygon(train_polygon, label_direction=1)
    polygon_utils.plot_polygon(train_y_coords, label_direction=-1)
    plt.draw()
    plt.pause(0.001)


def main(_):
    # Create the input placeholder
    x_image = tf.placeholder(tf.float32, [batch_size, input_res, input_res, input_channels])

    # Define loss and optimizer
    y_coords_ = tf.placeholder(tf.float32, [batch_size, output_vertex_count, 2])

    # Build the graph for the deep net
    # y_coords, keep_prob = model.polygon_regressor(x_image=x_image, input_res=input_res, input_channels=input_channels,
    #                                               encoding_length=encoding_length,
    #                                               output_vertex_count=output_vertex_count, weight_decay=weight_decay)

    y_coords, keep_prob = model.feature_extractor_polygon_regressor(x_image=x_image,
                                                                    feature_extractor_name=FEATURE_EXTRACTOR_PARAMS[
                                                                        "name"],
                                                                    encoding_length=ENCODING_LENGTH,
                                                                    output_vertex_count=output_vertex_count,
                                                                    weight_decay=weight_decay)

    # Build the objective loss function as well as the accuracy parts of the graph
    objective_loss, accuracy, accuracy2, accuracy3 = loss.loss_and_accuracy(y_coords_, y_coords, batch_size, correct_dist_threshold)
    tf.add_to_collection('losses', objective_loss)

    # Add all losses (objective loss + weigh loss for now)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    tf.summary.scalar('total_loss', total_loss)

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    learning_rate = tf.train.piecewise_constant(global_step, LEARNING_RATE_PARAMS["boundaries"],
                                                LEARNING_RATE_PARAMS["values"])
    feature_extractor_learning_rate = tf.train.piecewise_constant(global_step,
                                                                  FEATURE_EXTRACTOR_LEARNING_RATE_PARAMS["boundaries"],
                                                                  FEATURE_EXTRACTOR_LEARNING_RATE_PARAMS["values"])
    decoder_learning_rate = tf.train.piecewise_constant(global_step, DECODER_LEARNING_RATE_PARAMS["boundaries"],
                                                        DECODER_LEARNING_RATE_PARAMS["values"])

    # Choose the ensemble of variables to train
    trainable_variables = tf.trainable_variables()
    feature_extractor_variables = []
    decoder_variables = []
    other_variables = []
    for var in trainable_variables:
        if var.name.startswith(FEATURE_EXTRACTOR_PARAMS["name"]):
            feature_extractor_variables.append(var)
        elif var.name.startswith(DECODER_PARAMS["name"]):
            decoder_variables.append(var)
        else:
            other_variables.append(var)

    with tf.name_scope('adam_optimizer'):
        # This optimizer uses 3 different optimizers to allow different learning rates for the 3 different sub-graphs
        other_train_op = tf.train.AdamOptimizer(learning_rate)
        feature_extractor_train_op = tf.train.AdamOptimizer(feature_extractor_learning_rate)
        decoder_train_op = tf.train.AdamOptimizer(decoder_learning_rate)
        grads = tf.gradients(total_loss, other_variables + feature_extractor_variables + decoder_variables)
        feature_extractor_variable_count = len(feature_extractor_variables)
        other_variable_count = len(other_variables)
        other_grads = grads[:other_variable_count]
        feature_extractor_grads = grads[other_variable_count:other_variable_count + feature_extractor_variable_count]
        decoder_grads = grads[other_variable_count + feature_extractor_variable_count:]
        other_train_step = other_train_op.apply_gradients(zip(other_grads, other_variables))
        feature_extractor_train_step = feature_extractor_train_op.apply_gradients(zip(feature_extractor_grads, feature_extractor_variables))
        decoder_train_step = decoder_train_op.apply_gradients(zip(decoder_grads, decoder_variables))
        train_step = tf.group(other_train_step, feature_extractor_train_step, decoder_train_step, tf.assign_add(global_step, 1))

    # Summaries
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(LOGS_DIR, "train"), tf.get_default_graph())
    val_writer = tf.summary.FileWriter(os.path.join(LOGS_DIR, "val"), tf.get_default_graph())

    # Dataset
    train_dataset_filename = os.path.join(TFRECORDS_DIR, "train.tfrecord")
    train_images, train_polygons = dataset.read_and_decode(train_dataset_filename, input_res,
                                                           output_vertex_count, batch_size, INPUT_DYNAMIC_RANGE)
    val_dataset_filename = os.path.join(TFRECORDS_DIR, "val.tfrecord")
    val_images, val_polygons = dataset.read_and_decode(val_dataset_filename, input_res,
                                                       output_vertex_count, batch_size, INPUT_DYNAMIC_RANGE,
                                                       augment_dataset=False)

    # Savers
    feature_extractor_saver = tf.train.Saver(feature_extractor_variables)
    decoder_saver = tf.train.Saver(decoder_variables)
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
        else:
            # Else load pre-trained parts of the model as initialisation
            if FEATURE_EXTRACTOR_PARAMS["use_pretrained_weights"]:
                print("Initializing feature extractor variables from checkpoint {}".format(
                    FEATURE_EXTRACTOR_PARAMS["checkpoint_filepath"]))
                feature_extractor_saver.restore(sess, FEATURE_EXTRACTOR_PARAMS["checkpoint_filepath"])
            if DECODER_PARAMS["use_pretrained_weights"]:
                print("Initializing Decoder variables from directory {}".format(
                    DECODER_PARAMS["checkpoint_dir"]))
                checkpoint = tf.train.get_checkpoint_state(DECODER_PARAMS["checkpoint_dir"])
                if checkpoint and checkpoint.model_checkpoint_path:  # First check if the whole model has a checkpoint
                    print("Restoring checkpoint {}".format(checkpoint.model_checkpoint_path))
                    decoder_saver.restore(sess, checkpoint.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        init_plots()

        print("Model has {} trainable variables".format(
            tf_utils.count_number_trainable_params(trainable_variables=trainable_variables))
        )

        i = tf.train.global_step(sess, global_step)
        while i <= max_iter:
            train_image_batch, train_polygon_batch = sess.run([train_images, train_polygons])
            if i % train_loss_accuracy_steps == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                train_summary, _, train_loss, train_accuracy, train_accuracy2, train_accuracy3, train_y_coords = sess.run(
                    [merged_summaries, train_step, total_loss, accuracy, accuracy2, accuracy3, y_coords],
                    feed_dict={x_image: train_image_batch, y_coords_: train_polygon_batch,
                               keep_prob: dropout_keep_prob}, options=run_options, run_metadata=run_metadata)
                train_writer.add_summary(train_summary, i)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                print('step %d, training loss = %g, accuracy = %g accuracy2 = %g accuracy3 = %g' % (i, train_loss, train_accuracy, train_accuracy2, train_accuracy3))
                plot_results(1, train_image_batch, train_polygon_batch, train_y_coords)
            else:
                _ = sess.run([train_step], feed_dict={x_image: train_image_batch, y_coords_: train_polygon_batch,
                                                      keep_prob: dropout_keep_prob})

            # Measure validation loss and accuracy
            if i % val_loss_accuracy_steps == 1:
                val_image_batch, val_polygon_batch = sess.run([val_images, val_polygons])
                val_summary, val_loss, val_accuracy, val_accuracy2, val_accuracy3, val_y_coords = sess.run(
                    [merged_summaries, total_loss, accuracy, accuracy2, accuracy3, y_coords],
                    feed_dict={
                        x_image: val_image_batch,
                        y_coords_: val_polygon_batch, keep_prob: 1.0})
                val_writer.add_summary(val_summary, i)

                print('step %d, validation loss = %g, accuracy = %g, accuracy2 = %g, accuracy3 = %g' % (i, val_loss, val_accuracy, val_accuracy2, val_accuracy3))
                plot_results(2, val_image_batch, val_polygon_batch, val_y_coords)

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
    tf.app.run(main=main)
