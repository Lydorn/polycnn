from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

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
input_res = 67
input_channels = 3
INPUT_DYNAMIC_RANGE = [-1, 1]  # [min, max] the network expects
output_vertex_count = 4
TFRECORDS_DIR = os.path.join(ROOT_DIR, "data/photovoltaic_array_location_dataset/tfrecords.polycnn")

# Model
ENCODING_LENGTH = 128
FEATURE_EXTRACTOR_PARAMS = {
    "name": "InceptionV4"
}
model_name = "polygonize-photovoltaic-arrays"
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "code/polycnn/runs/current/checkpoints")

# Inference
batch_size = 256
correct_dist_threshold = 1 / input_res

# Validation
# The following value has to be equal to batch_size because the effective size of any batch has to be equal to
# batch_size (the custom loss function needs this condition)
dataset_test_size = 256

# Outputs
SAVE_DIR = os.path.join(ROOT_DIR, "code/polycnn/pred")


# --- --- #

def save_result(train_image, train_polygon, train_y_coords, filepath):
    # Image with polygons overlaid
    im_res = train_image.shape[0]
    image = (train_image - INPUT_DYNAMIC_RANGE[0]) / (INPUT_DYNAMIC_RANGE[1] - INPUT_DYNAMIC_RANGE[0])
    train_polygon = train_polygon * im_res
    train_y_coords = train_y_coords * im_res
    plt.cla()
    fig = plt.imshow(image)
    polygon_utils.plot_polygon(train_polygon, color="#28ff0288", draw_labels=False)
    polygon_utils.plot_polygon(train_y_coords, color="#ff8800", draw_labels=False)
    plt.margins(0)
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(filepath + ".png", bbox_inches='tight', pad_inches=0)
    # Save polygons
    train_polygon_array = np.array(train_polygon, dtype=np.float16)
    train_y_coords_array = np.array(train_y_coords, dtype=np.float16)
    np.save(filepath + ".gt.npy", train_polygon_array)
    np.save(filepath + ".pred.npy", train_y_coords_array)


def save_results(train_image_batch, train_polygon_batch, train_y_coords_batch, save_dir):
    batch_size = train_image_batch.shape[0]
    for i in range(batch_size):
        print("Plotting {}/{}".format(i + 1, batch_size))
        save_result(train_image_batch[i], train_polygon_batch[i], train_y_coords_batch[i],
                    os.path.join(save_dir, "image_polygons.{:04d}".format(i)))


def main(_):
    # Create the input placeholder
    x_image = tf.placeholder(tf.float32, [batch_size, input_res, input_res, input_channels])

    # Define loss and optimizer
    y_coords_ = tf.placeholder(tf.float32, [batch_size, output_vertex_count, 2])

    y_coords, keep_prob = model.feature_extractor_polygon_regressor(x_image=x_image,
                                                                    feature_extractor_name=FEATURE_EXTRACTOR_PARAMS[
                                                                        "name"],
                                                                    encoding_length=ENCODING_LENGTH,
                                                                    output_vertex_count=output_vertex_count)

    # Build the objective loss function as well as the accuracy parts of the graph
    _, accuracy, accuracy2, accuracy3 = loss.loss_and_accuracy(y_coords_, y_coords, batch_size, correct_dist_threshold)

    # Dataset
    test_dataset_filename = os.path.join(TFRECORDS_DIR, "test.tfrecord")
    test_images, test_polygons = dataset.read_and_decode(test_dataset_filename, input_res,
                                                         output_vertex_count, batch_size, INPUT_DYNAMIC_RANGE,
                                                         augment_dataset=False)

    # Saver
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore checkpoint if one exists
        checkpoint = tf.train.get_checkpoint_state(CHECKPOINTS_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:  # First check if the whole model has a checkpoint
            print("Restoring {} checkpoint {}".format(model_name, checkpoint.model_checkpoint_path))
            saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            print("No checkpoint was found, exiting...")
            exit()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        test_image_batch, test_polygon_batch = sess.run([test_images, test_polygons])
        test_accuracy, test_accuracy2, test_accuracy3, test_y_coords = sess.run(
            [accuracy, accuracy2, accuracy3, y_coords],
            feed_dict={
                x_image: test_image_batch,
                y_coords_: test_polygon_batch, keep_prob: 1.0})

        print('Test accuracy = %g accuracy2 = %g accuracy3 = %g' % (test_accuracy, test_accuracy2, test_accuracy3))

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        save_results(test_image_batch, test_polygon_batch, test_y_coords, SAVE_DIR)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run(main=main)
