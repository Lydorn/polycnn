import tensorflow as tf
import os.path
import skimage.io as io

# --- Params --- #

ROOT_DIR = "../../"

SEED = 0

# --- --- #


def read_and_decode(tfrecord_filename, im_res, vertex_count, batch_size, dynamic_range, seed=None):
    filename_queue = tf.train.string_input_producer(
        [tfrecord_filename])

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'polygon_raw': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    polygon = tf.decode_raw(features['polygon_raw'], tf.float16)

    image_shape = tf.stack([im_res, im_res, 1])
    polygon_shape = tf.stack([vertex_count, 2])

    image = tf.reshape(image, image_shape)
    polygon = tf.reshape(polygon, polygon_shape)

    # Normalize data
    image = (image / 255) * (dynamic_range[1] - dynamic_range[0]) + dynamic_range[0]
    polygon = polygon / im_res

    min_queue_examples = 256
    images, polygons = tf.train.shuffle_batch([image, polygon],
                                              batch_size=batch_size,
                                              capacity=min_queue_examples + 3 * batch_size,
                                              num_threads=2,
                                              min_after_dequeue=min_queue_examples,
                                              allow_smaller_final_batch=False,
                                              seed=seed)
    return images, polygons


if __name__ == "__main__":
    # --- Params --- #
    im_res = 64
    vertex_count = 4
    batch_size = 64
    dynamic_range = [-1, 1]

    tfrecord_filename = os.path.join(ROOT_DIR, "data/polygons/encoder_decoder/train.tfrecord")
    # --- --- #

    # Even when reading in multiple threads, share the filename
    # queue.
    image, polygon = read_and_decode(tfrecord_filename, im_res, vertex_count, batch_size, dynamic_range, seed=SEED)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Let's read off 3 batches just for example
        for i in range(3):
            img, poly = sess.run([image, polygon])
            print(img[0, :, :, :].shape)

            print('current batch')
            io.imshow(img[0][:, :, 0])
            io.show()

            print(poly[0])

        coord.request_stop()
        coord.join(threads)
