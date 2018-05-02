from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import os
import numpy as np
import skimage.morphology

sys.path.append("../utils/")
import polygon_utils

import model
import dataset
import loss

FLAGS = None

# --- Params --- #
INSIDE_DOCKER = False

if not INSIDE_DOCKER:
    import matplotlib.pyplot as plt
    ROOT_DIR = "../../"
else:
    ROOT_DIR = "/workspace"

# Data
input_res = 64
input_channels = 3
INPUT_DYNAMIC_RANGE = [-1, 1]  # [min, max] the network expects
output_vertex_count = 4
TFRECORDS_DIR = os.path.join(ROOT_DIR, "data/photovoltaic_array_location_dataset/tfrecords.unet_and_vectorization")

# Model
model_name = "unet-and-vectorization-photovoltaic-arrays"
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "code/unet_and_vectorization/runs/current/checkpoints")

# Inference
batch_size = 256
correct_dist_threshold = 1 / input_res

# Validation
# The following value has to be equal to batch_size because the effective size of any batch has to be equal to
# batch_size (the custom loss function needs this condition)
dataset_test_size = 256  # TODO: Remove this constraint to allow validation on more that 1 batch (very low priority)

# Outputs
SAVE_DIR = os.path.join(ROOT_DIR, "code/unet_and_vectorization/pred")


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
    polygon_utils.plot_polygon(train_y_coords, color="#ff0000", draw_labels=False)
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
    y_image_ = tf.placeholder(tf.float32, [batch_size, input_res, input_res, 1])

    y_image, mode_training = model.make_unet(x_image=x_image)

    total_loss = loss.cross_entropy(y_image, y_image_)

    # Dataset
    test_dataset_filename = os.path.join(TFRECORDS_DIR, "test.tfrecord")
    test_images, test_polygons, test_raster_polygons = dataset.read_and_decode(test_dataset_filename, input_res,
                                                                               output_vertex_count, batch_size,
                                                                               INPUT_DYNAMIC_RANGE,
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

        test_image_batch, test_polygon_batch, test_raster_polygon_batch = sess.run([test_images, test_polygons, test_raster_polygons])
        test_loss, test_y_image_batch = sess.run(
            [total_loss, y_image],
            feed_dict={
                x_image: test_image_batch,
                y_image_: test_raster_polygon_batch, mode_training: True
            })

        print("Test loss= {}".format(test_loss))

        # Threshold output
        test_raster_polygon_batch = 0.5 < test_raster_polygon_batch
        test_y_image_batch = 0.5 < test_y_image_batch

        # Polygonize
        print("Polygonizing...")
        y_coord_batch_list = []
        for test_raster_polygon, test_y_image in zip(test_raster_polygon_batch, test_y_image_batch):
            test_raster_polygon = test_raster_polygon[:, :, 0]
            test_y_image = test_y_image[:, :, 0]

            # Select only one blob
            seed = np.logical_and(test_raster_polygon, test_y_image)
            test_y_image = skimage.morphology.reconstruction(seed, test_y_image, method='dilation', selem=None, offset=None)

            # Vectorize
            test_y_coords = polygon_utils.raster_to_polygon(test_y_image, output_vertex_count)
            y_coord_batch_list.append(test_y_coords)
        y_coord_batch = np.array(y_coord_batch_list)

        # Normalize
        y_coord_batch = y_coord_batch / input_res

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        save_results(test_image_batch, test_polygon_batch, y_coord_batch, SAVE_DIR)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run(main=main)
