import numpy as np
from PIL import Image, ImageDraw
import os
import tensorflow as tf
import sys

sys.path.append("../utils/")
import polygon_utils


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# --- Params --- #

ROOT_DIR = "../../"

im_res = 64
vertex_count = 4
dataset_train_size = 100000
dataset_val_size = 64
dataset_test_size = 64
dataset_directory_path = os.path.join(ROOT_DIR, "data/polygon_encoder_decoder")

# --- --- #


def generate_polygon_data(dataset_size, fold="train"):
    writer = tf.python_io.TFRecordWriter(os.path.join(dataset_directory_path, fold + ".tfrecord"))
    for i in range(dataset_size):
        vertex_list = polygon_utils.generate_polygon(cx=im_res / 2, cy=im_res / 2, ave_radius=im_res * 0.25 * 0.9,
                                                     irregularity=0.5,
                                                     spikeyness=0.5, vertex_count=vertex_count)

        im = Image.new('1', (im_res, im_res))
        im_px_access = im.load()
        draw = ImageDraw.Draw(im)

        # either use .polygon(), if you want to fill the area with a solid colour
        draw.polygon(vertex_list, fill=1)

        # Save

        im_raw = np.array(im).tostring()
        vertex_array = np.array(vertex_list)
        vertex_array = vertex_array.astype(np.float16)  # We do not need 64bit precision
        vertex_array_raw = vertex_array.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(im_raw),
            'polygon_raw': _bytes_feature(vertex_array_raw)}))

        writer.write(example.SerializeToString())

        # filename = os.path.join(dataset_directory_path, fold, filename_format.format(i) + ".png")
        # im.save(filename)
        #
        # vertex_array = np.array(vertex_list)
        # filename = os.path.join(dataset_directory_path, fold, filename_format.format(i) + ".npy")
        # np.save(filename, vertex_array)

    writer.close()


if __name__ == "__main__":
    if not os.path.exists(dataset_directory_path):
        os.makedirs(dataset_directory_path)

    print("Generating train set...")
    generate_polygon_data(dataset_train_size, fold="train")
    print("Generating validation set...")
    generate_polygon_data(dataset_val_size, fold="val")
    print("Generating test set...")
    generate_polygon_data(dataset_test_size, fold="test")
