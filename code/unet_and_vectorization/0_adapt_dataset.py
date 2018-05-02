import sys
import os
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

# --- Params --- #
DEBUG = True

INPUT_TFRECORDS_DIR = \
    "../../data/photovoltaic_array_location_dataset/tfrecords.polycnn"
OUTPUT_TFRECORDS_DIR = \
    "../../data/photovoltaic_array_location_dataset/tfrecords.unet_and_vectorization"

# --- --- #


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def add_raster_polygon(input_tfrecord, output_tfrecord):
    # Save tfrecords
    filtered_writer = tf.python_io.TFRecordWriter(output_tfrecord)
    example = tf.train.Example()
    for i, record in enumerate(tf.python_io.tf_record_iterator(input_tfrecord)):
        example.ParseFromString(record)
        f = example.features.feature
        width = f['width'].int64_list.value[0]
        height = f['height'].int64_list.value[0]
        polygon = np.fromstring(f['polygon_raw'].bytes_list.value[0], dtype=np.float16)
        polygon = polygon.reshape([-1, 2])

        # Compute rasterization of polygon
        raster_polygon = Image.new('1', (width, height))
        im_px_access = raster_polygon.load()
        draw = ImageDraw.Draw(raster_polygon)
        vertex_list = [(vertex[0], vertex[1]) for vertex in polygon]
        draw.polygon(vertex_list, fill=1)
        raster_polygon_raw = np.array(raster_polygon).tostring()

        # Save new tfrecord
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': f['image_raw'],
            'width': f['width'],
            'height': f['height'],
            'polygon_raw': f['polygon_raw'],
            'raster_polygon_raw': _bytes_feature(raster_polygon_raw)
        }))

        filtered_writer.write(example.SerializeToString())

    filtered_writer.close()


def main():
    if not os.path.exists(OUTPUT_TFRECORDS_DIR):
        os.makedirs(OUTPUT_TFRECORDS_DIR)

    for fold in ["train", "val", "test"]:
        input_tfrecord = os.path.join(INPUT_TFRECORDS_DIR, fold + ".tfrecord")
        output_tfrecord = os.path.join(OUTPUT_TFRECORDS_DIR, fold + ".tfrecord")
        add_raster_polygon(input_tfrecord, output_tfrecord)


if __name__ == "__main__":
    main()
