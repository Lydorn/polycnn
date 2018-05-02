import math
import random
import numpy as np
import scipy.spatial
from shapely import geometry
import skimage

import python_utils

if python_utils.module_exists("matplotlib.pyplot"):
    import matplotlib.pyplot as plt
    from skimage.measure import approximate_polygon


def is_polygon_clockwise(polygon):
    rolled_polygon = np.roll(polygon, shift=1, axis=0)
    double_signed_area = np.sum((rolled_polygon[:, 0] - polygon[:, 0])*(rolled_polygon[:, 1] + polygon[:, 1]))
    if 0 < double_signed_area:
        return True
    else:
        return False


def orient_polygon(polygon, orientation="CW"):
    poly_is_orientated_cw = is_polygon_clockwise(polygon)
    if (poly_is_orientated_cw and orientation == "CCW") or (not poly_is_orientated_cw and orientation == "CW"):
        return np.flip(polygon, axis=0)
    else:
        return polygon


def raster_to_polygon(image, vertex_count):
    contours = skimage.measure.find_contours(image, 0.5)
    contour = np.empty_like(contours[0])
    contour[:, 0] = contours[0][:, 1]
    contour[:, 1] = contours[0][:, 0]

    # Simplify until vertex_count
    tolerance = 0.1
    tolerance_step = 0.1
    simplified_contour = contour
    while 1 + vertex_count < len(simplified_contour):
        simplified_contour = approximate_polygon(contour, tolerance=tolerance)
        tolerance += tolerance_step

    simplified_contour = simplified_contour[:-1]

    # plt.imshow(image, cmap="gray")
    # plot_polygon(simplified_contour, draw_labels=False)
    # plt.show()

    return simplified_contour


def l2diffs(polygon1, polygon2):
    """
    Computes vertex-wise L2 difference between the two polygons.
    As the two polygons may not have the same starting vertex,
    all shifts are considred and the shift resulting in the minimum mean L2 difference is chosen
    
    :param polygon1: 
    :param polygon2: 
    :return: 
    """
    # Make polygons of equal length
    if len(polygon1) != len(polygon2):
        while len(polygon1) < len(polygon2):
            polygon1 = np.append(polygon1, [polygon1[-1, :]], axis=0)
        while len(polygon2) < len(polygon1):
            polygon2 = np.append(polygon2, [polygon2[-1, :]], axis=0)
    vertex_count = len(polygon1)

    def naive_l2diffs(polygon1, polygon2):
        naive_l2diffs_result = np.sqrt(np.power(np.sum(polygon1 - polygon2, axis=1), 2))
        return naive_l2diffs_result

    min_l2_diffs = naive_l2diffs(polygon1, polygon2)
    min_mean_l2_diffs = np.mean(min_l2_diffs, axis=0)
    for i in range(1, vertex_count):
        current_naive_l2diffs = naive_l2diffs(np.roll(polygon1, shift=i, axis=0), polygon2)
        current_naive_mean_l2diffs = np.mean(current_naive_l2diffs, axis=0)
        if current_naive_mean_l2diffs < min_mean_l2_diffs:
            min_l2_diffs = current_naive_l2diffs
            min_mean_l2_diffs = current_naive_mean_l2diffs
    return min_l2_diffs


def polygon_iou(polygon1, polygon2):
    poly1 = geometry.Polygon(polygon1).buffer(0)
    poly2 = geometry.Polygon(polygon2).buffer(0)
    intersection_poly = poly1.intersection(poly2)
    union_poly = poly1.union(poly2)
    intersection_area = intersection_poly.area
    union_area = union_poly.area
    if union_area:
        iou = intersection_area / union_area
    else:
        iou = 0
    return iou


def generate_polygon(cx, cy, ave_radius, irregularity, spikeyness, vertex_count):
    """
    Start with the centre of the polygon at cx, cy,
    then creates the polygon by sampling points on a circle around the centre.
    Random noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    cx, cy - coordinates of the "centre" of the polygon
    ave_radius - in px, the average radius of this polygon, this roughly controls how large the polygon is,
        really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to
        [0, 2 * pi / vertex_count]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius ave_radius.
        [0,1] will map to [0, ave_radius]
    vertex_count - self-explanatory

    Returns a list of vertices, in CCW order.
    """

    irregularity = clip(irregularity, 0, 1) * 2 * math.pi / vertex_count
    spikeyness = clip(spikeyness, 0, 1) * ave_radius

    # generate n angle steps
    angle_steps = []
    lower = (2 * math.pi / vertex_count) - irregularity
    upper = (2 * math.pi / vertex_count) + irregularity
    angle_sum = 0
    for i in range(vertex_count):
        tmp = random.uniform(lower, upper)
        angle_steps.append(tmp)
        angle_sum = angle_sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = angle_sum / (2 * math.pi)
    for i in range(vertex_count):
        angle_steps[i] = angle_steps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(vertex_count):
        r_i = clip(random.gauss(ave_radius, spikeyness), 0, 2 * ave_radius)
        x = cx + r_i * math.cos(angle)
        y = cy + r_i * math.sin(angle)
        points.append((x, y))

        angle = angle + angle_steps[i]

    return points


def clip(x, mini, maxi):
    if mini > maxi:
        return x
    elif x < mini:
        return mini
    elif x > maxi:
        return maxi
    else:
        return x


def compute_bounding_box(polygon, scale=1, boundingbox_margin=0, fit=None):
    # Compute base bounding box
    bounding_box = [np.min(polygon[:, 0]), np.min(polygon[:, 1]), np.max(polygon[:, 0]), np.max(polygon[:, 1])]
    # Scale
    half_width = math.ceil((bounding_box[2] - bounding_box[0]) * scale / 2)
    half_height = math.ceil((bounding_box[3] - bounding_box[1]) * scale / 2)
    # Add margin
    half_width += boundingbox_margin
    half_height += boundingbox_margin
    # Compute square bounding box
    if fit == "square":
        half_width = half_height = max(half_width, half_height)
    center = [round((bounding_box[0] + bounding_box[2]) / 2), round((bounding_box[1] + bounding_box[3]) / 2)]
    bounding_box = [center[0] - half_width, center[1] - half_height, center[0] + half_width, center[1] + half_height]
    return bounding_box


def compute_patch(polygon, patch_size):
    centroid = np.mean(polygon, axis=0)
    half_height = half_width = patch_size / 2
    bounding_box = [math.ceil(centroid[0] - half_width), math.ceil(centroid[1] - half_height),
                    math.ceil(centroid[0] + half_width), math.ceil(centroid[1] + half_height)]
    return bounding_box


def bounding_box_within_bounds(bounding_box, bounds):
    return bounds[0] <= bounding_box[0] and bounds[1] <= bounding_box[1] and bounding_box[2] <= bounds[2] and \
           bounding_box[3] <= bounds[3]


def bounding_box_area(bounding_box):
    return (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1])


def convert_to_image_patch_space(polygon_image_space, bounding_box):
    polygon_image_patch_space = np.empty_like(polygon_image_space)
    polygon_image_patch_space[:, 0] = polygon_image_space[:, 0] - bounding_box[0]
    polygon_image_patch_space[:, 1] = polygon_image_space[:, 1] - bounding_box[1]
    return polygon_image_patch_space


def strip_redundant_vertex(vertices, epsilon=1):
    new_vertices = vertices
    if 1 < vertices.shape[0]:
        if np.sum(np.absolute(vertices[0, :] - vertices[-1, :])) < epsilon:
            new_vertices = vertices[:-1, :]
    return new_vertices


def simplify_polygon(polygon, tolerance=1):
    approx_polygon = approximate_polygon(polygon, tolerance=tolerance)
    return approx_polygon


def compute_diameter(polygon):
    dist = scipy.spatial.distance.cdist(polygon, polygon)
    return dist.max()


def plot_polygon(polygon, color=None, draw_labels=True, label_direction=1):
    polygon_closed = np.append(polygon, [polygon[0, :]], axis=0)
    plt.plot(polygon_closed[:, 0], polygon_closed[:, 1], color=color, linewidth=3.0)

    if draw_labels:
        labels = range(1, polygon.shape[0] + 1)
        for label, x, y in zip(labels, polygon[:, 0], polygon[:, 1]):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20 * label_direction, 20 * label_direction),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.25', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


if __name__ == "__main__":
    polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    poly_clockwise = is_polygon_clockwise(polygon)
    print("poly_clockwise: {}".format(poly_clockwise))

    polygon = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    poly_clockwise = is_polygon_clockwise(polygon)
    print("poly_clockwise: {}".format(poly_clockwise))

    polygon = orient_polygon(polygon, orientation="CW")
    poly_clockwise = is_polygon_clockwise(polygon)
    print("poly_clockwise: {}".format(poly_clockwise))
