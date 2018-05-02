import sys
from os import path
import tensorflow as tf
import numpy as np

sys.path.append("../utils/")
import polygon_utils

# --- Params --- #
DEBUG = True

TFRECORDS_DIR = \
    "../../data/photovoltaic_array_location_dataset/tfrecords.polycnn"

MAX_DIAMETER = 67 * 0.8  # Leave at least 20% context around polygon

# --- --- #


def print_debug(*args):
    if DEBUG:
        for arg in args:
            print(arg)


def filter_tfrecord(original_tfrecord, filtered_tfrecord, max_diameter):
    # Get total number of records
    record_count = 0
    for _ in tf.python_io.tf_record_iterator(original_tfrecord):
        record_count += 1

    # Save tfrecords
    filtered_writer = tf.python_io.TFRecordWriter(filtered_tfrecord)
    example = tf.train.Example()
    original_record_count = 0
    filtered_record_count = 0
    for i, record in enumerate(tf.python_io.tf_record_iterator(original_tfrecord)):
        original_record_count += 1
        example.ParseFromString(record)
        f = example.features.feature
        polygon = np.fromstring(f['polygon_raw'].bytes_list.value[0], dtype=np.float16)
        polygon = polygon.reshape([-1, 2])
        if polygon_utils.compute_diameter(polygon) < max_diameter:
            filtered_record_count += 1
            filtered_writer.write(record)

    filtered_writer.close()
    print("Original record count = {}".format(original_record_count))
    print("Filtered record count = {}".format(filtered_record_count))


def main():
    original_tfrecord = path.join(TFRECORDS_DIR, "dataset.tfrecord")
    filtered_tfrecord = path.join(TFRECORDS_DIR, "filtered_dataset.tfrecord")
    filter_tfrecord(original_tfrecord, filtered_tfrecord, MAX_DIAMETER)


if __name__ == "__main__":
    main()
