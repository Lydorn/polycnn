import tensorflow as tf
import sys
import math
import numpy as np

sys.path.append("../")
import polygon_utils
import python_utils


if python_utils.module_exists("skimage.io"):
    import skimage.io as io

# --- Params --- #

SEED = 0

RANDOM_CROP_AMPLITUDE = 5

# --- --- #


def polygon_flip_up_down(polygon, vertex_count, im_res):
    invert_mat = np.array([[1, 0], [0, -1]], dtype=np.float16)
    translate_mat = np.tile(np.array([0, im_res]), (vertex_count, 1))
    polygon = tf.add(tf.matmul(polygon, invert_mat), translate_mat)
    # Re-orient polygon
    polygon = tf.reverse(polygon, axis=[0])
    return polygon


def read_and_decode(tfrecords_filename, im_res, vertex_count, batch_size, dynamic_range, augment_dataset=True):
    filename_queue = tf.train.string_input_producer(
        [tfrecords_filename])

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'polygon_raw': tf.FixedLenFeature([], tf.string)
        })
    image_flat = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int64)
    width = tf.cast(features['width'], tf.int64)
    polygon_flat = tf.decode_raw(features['polygon_raw'], tf.float16)

    # Reshape tensors
    image_shape = tf.stack([height, width, 3])
    polygon_shape = tf.stack([vertex_count, 2])
    image = tf.reshape(image_flat, image_shape)
    polygon = tf.reshape(polygon_flat, polygon_shape)

    # max_width_height = tf.maximum(width, height)
    # new_width = (width / max_width_height) * im_res
    # new_height = (height / max_width_height) * im_res
    # new_shape = tf.stack([new_height, new_width])

    # image = tf.image.resize_images(images=image, size=[im_res, im_res])
    # image = image / 255

    if augment_dataset:
        # Apply random rotation to image
        angle = tf.random_uniform([], maxval=2*math.pi, dtype=tf.float32, seed=SEED)
        image = tf.contrib.image.rotate(image, angle, interpolation='BILINEAR')
    # Apply crop to image
    if augment_dataset:
        crop_res = im_res + RANDOM_CROP_AMPLITUDE
    else:
        crop_res = im_res
    image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                   target_height=crop_res,
                                                   target_width=crop_res)

    # Apply crop to polygon
    polygon = polygon + [(crop_res - tf.cast(width, tf.float16)) / 2, (crop_res - tf.cast(height, tf.float16)) / 2]
    if augment_dataset:
        # Apply rotation to polygon
        center = [crop_res / 2, crop_res / 2]
        polygon = polygon - center
        rot_mat = tf.cast(tf.stack([(tf.cos(angle), -tf.sin(angle)), (tf.sin(angle), tf.cos(angle))], axis=0), tf.float16)
        polygon = tf.matmul(polygon, rot_mat)
        polygon = polygon + center

        # Apply random flip
        flip = tf.random_uniform([], dtype=tf.float16, seed=SEED)
        image, polygon = tf.cond(0.5 <= flip,
                                 lambda: (tf.image.flip_up_down(image), polygon_flip_up_down(polygon, vertex_count, crop_res)),
                                 lambda: (image, polygon))

        # Apply random crop
        if RANDOM_CROP_AMPLITUDE:
            crop_offset = tf.random_uniform([2], maxval=RANDOM_CROP_AMPLITUDE, dtype=tf.int32, seed=SEED)
            image = tf.image.crop_to_bounding_box(
                image,
                crop_offset[1],
                crop_offset[0],
                im_res,
                im_res
            )
            polygon = polygon - tf.cast(crop_offset, dtype=tf.float16)

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
                                              seed=SEED)
    return images, polygons


if __name__ == "__main__":
    # --- Params --- #
    tfrecord_filepath = "/home/nigirard/epitome-polygon-deep-learning/data/photovoltaic_array_location_dataset/tfrecords/" \
                        "train.tfrecord"
    im_res = 67
    vertex_count = 4
    batch_size = 256
    dynamic_range = [-1, 1]
    augment_dataset = True
    # --- --- #

    # Even when reading in multiple threads, share the filename
    # queue.
    image, polygon = read_and_decode(tfrecord_filepath, im_res, vertex_count, batch_size, dynamic_range,
                                     augment_dataset=augment_dataset)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Let's read off 3 batches just for example
        for i in range(1):
            imgs, polys = sess.run([image, polygon])
            print(imgs.min(), imgs.max())
            print(polys.min(), polys.max())

            for img, poly in zip(imgs, polys):
                img = (img - dynamic_range[0]) / (dynamic_range[1] - dynamic_range[0])
                poly = poly * im_res
                io.imshow(img)
                polygon_utils.plot_polygon(poly, label_direction=1)
                io.show()

        coord.request_stop()
        coord.join(threads)
