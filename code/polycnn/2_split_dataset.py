from os import path
import random

import tensorflow as tf


# --- Params --- #
DEBUG = True

SEED = 0

TFRECORDS_DIR = \
    "../../data/photovoltaic_array_location_dataset/tfrecords.polycnn"

VAL_SIZE = 256
TEST_SIZE = 256

# --- --- #


def print_debug(*args):
    if DEBUG:
        for arg in args:
            print(arg)


def split_tfrecord(original_tfrecord, train_tfrecord, val_tfrecord, val_size, test_tfrecord, test_size):
    # Get total number of records
    record_count = 0
    for record in tf.python_io.tf_record_iterator(original_tfrecord):
        record_count += 1

    # Shuffle
    shuffle = [x for x in range(record_count)]
    random.shuffle(shuffle)

    # Save tfrecords
    test_writer = tf.python_io.TFRecordWriter(test_tfrecord)
    val_writer = tf.python_io.TFRecordWriter(val_tfrecord)
    train_writer = tf.python_io.TFRecordWriter(train_tfrecord)
    for i, record in enumerate(tf.python_io.tf_record_iterator(original_tfrecord)):
        shuffled_i = shuffle[i]
        if shuffled_i < test_size:
            test_writer.write(record)
        elif shuffled_i < test_size + val_size:
            val_writer.write(record)
        else:
            train_writer.write(record)

    test_writer.close()
    val_writer.close()
    train_writer.close()


def main():
    random.seed(SEED)

    original_tfrecord = path.join(TFRECORDS_DIR, "filtered_dataset.tfrecord")
    train_tfrecord = path.join(TFRECORDS_DIR, "train.tfrecord")
    val_tfrecord = path.join(TFRECORDS_DIR, "val.tfrecord")
    test_tfrecord = path.join(TFRECORDS_DIR, "test.tfrecord")
    split_tfrecord(original_tfrecord, train_tfrecord, val_tfrecord, VAL_SIZE, test_tfrecord, TEST_SIZE)


if __name__ == "__main__":
    main()
