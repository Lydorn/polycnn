import sys
import os
import json
import math

from PIL import Image
import numpy as np
import tensorflow as tf

sys.path.append("../utils/")
import polygon_utils

# --- Params --- #
DEBUG = True

ROOT_DIR = "../../"

DATASET_DIR = os.path.join(ROOT_DIR, "data/photovoltaic_array_location_dataset")
GT_FILEPATH = os.path.join(DATASET_DIR, "gt/SolarArrayPolygons.json")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")

VERTEX_COUNT = 4

# If extracting bounding box:
BOUNDING_BOX_SCALE = 1.5  # Add 50%
BOUNDING_BOX_MARGIN = 16  # 16px on both sides
MIN_BOUNDING_BOX_AREA = 40 * 40  # Minimum area in pixels of the final bounding box (includes margin)

# If extracting constant patch:
RANDOM_CROP_AMPLITUDE = 5
PATCH_SIZE = 67 * math.sqrt(2) + RANDOM_CROP_AMPLITUDE  # Account for the diagonal length when rotating followed by a random crop

TFRECORDS_DIR = os.path.join(DATASET_DIR, "tfrecords.polycnn")


# --- --- #

def print_debug(*args):
    if DEBUG:
        for arg in args:
            print(arg)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def process_photovoltaic_array_data(gt_filepath, images_dir, vertex_count, tfrecord_filepath, boundingbox_margin=None,
                                    min_boundingbox_area=None):
    with open(gt_filepath) as f:
        data = json.load(f)
        polygons = data["polygons"]

        writer = tf.python_io.TFRecordWriter(tfrecord_filepath)

        polygon_count = len(polygons)
        last_image_name = None
        image = None
        image_bounds = None
        for i, polygon in enumerate(polygons):
            polygon_id = polygon["polygon_id"]
            city = polygon["city"].lower()
            image_name = polygon["image_name"]

            # If image changed, load new image
            if image_name != last_image_name:
                print("Processing city {}, image {}, polygon {}. Progression: {}/{}"
                      .format(city, image_name, polygon_id, i + 1, polygon_count))
                last_image_name = image_name  # Update for next polygon
                image = Image.open(os.path.join(images_dir, city, image_name + ".tif"))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image_bounds = [0, 0, image.size[0], image.size[1]]

            # Continue if shape is at least a triangle
            if 2 < len(polygon["polygon_vertices_pixels"]):
                vertices_array = np.array(polygon["polygon_vertices_pixels"])
                # In case start and end vertices are the same:
                vertices_array = polygon_utils.strip_redundant_vertex(vertices_array)
                vertices_array = polygon_utils.simplify_polygon(vertices_array, tolerance=1)
                if vertices_array.shape[0] == vertex_count:
                    # bounding_box = polygon_utils.compute_bounding_box(vertices_array,
                    #                                                   boundingbox_margin=boundingbox_margin)
                    bounding_box = polygon_utils.compute_patch(vertices_array, PATCH_SIZE)
                    if polygon_utils.bounding_box_within_bounds(bounding_box, image_bounds) \
                            and min_boundingbox_area <= polygon_utils.bounding_box_area(bounding_box):
                        polygon_image_patch_space = polygon_utils.convert_to_image_patch_space(vertices_array,
                                                                                               bounding_box)
                        polygon_image_patch_space = polygon_utils.orient_polygon(polygon_image_patch_space,
                                                                                 orientation="CW")
                        polygon_image_patch_space = polygon_image_patch_space.astype(
                            np.float16)  # We do not need 64bit precision
                        image_patch = image.crop(bounding_box)
                        width, height = image_patch.size
                        image_patch_raw = np.array(image_patch).tostring()
                        polygon_raw = polygon_image_patch_space.tostring()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'image_raw': _bytes_feature(image_patch_raw),
                            'width': _int64_feature(width),
                            'height': _int64_feature(height),
                            'polygon_raw': _bytes_feature(polygon_raw)}))

                        writer.write(example.SerializeToString())
        writer.close()


def main():
    if not os.path.exists(TFRECORDS_DIR):
        os.makedirs(TFRECORDS_DIR)

    tfrecord_filepath = os.path.join(TFRECORDS_DIR, "dataset.tfrecord")
    process_photovoltaic_array_data(GT_FILEPATH, IMAGES_DIR, VERTEX_COUNT, tfrecord_filepath,
                                    boundingbox_margin=BOUNDING_BOX_MARGIN, min_boundingbox_area=MIN_BOUNDING_BOX_AREA)


if __name__ == "__main__":
    main()
