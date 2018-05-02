import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../utils/")
import polygon_utils


# --- Params --- #
ROOT_DIR = "../../"

IM_RES = 64

FIGURE_FILENAME = "cumulative_accuracy"

# --- ---


def compute_stats(pred_dir, method_name, curve_color):
    filename_bases = [f.strip(".gt.npy") for f in os.listdir(pred_dir) if f.endswith(".gt.npy")]
    filename_bases_count = len(filename_bases)

    mean_iou = 0
    all_l2diffs = []
    for i, f in enumerate(filename_bases):
        # print("Processing for prediction : {0}    progression  : {1}/{2}".format(f, i + 1, filename_bases_count))
        polygon1 = np.load(os.path.join(pred_dir, f + ".gt.npy"))
        polygon2 = np.load(os.path.join(pred_dir, f + ".pred.npy"))

        # Accuracy
        l2diffs = polygon_utils.l2diffs(polygon1, polygon2)
        # print("l2diffs = {}".format(l2diffs))
        all_l2diffs.append(l2diffs)

        # IoU
        iou = polygon_utils.polygon_iou(polygon1, polygon2)
        # print("IoU = {}".format(iou))
        mean_iou += iou

    mean_iou /= filename_bases_count
    print("Mean IoU = {}".format(mean_iou))

    # Accuracy
    all_l2diffs = np.array(all_l2diffs)
    all_l2diffs = all_l2diffs.flatten()
    bins = np.arange(0, IM_RES // 2, 0.5)
    l2diff_histogram = np.histogram(all_l2diffs, bins=bins)

    # plt.hist(l2diff_histogram[0], bins=bins)
    # plt.show()

    cumulation = np.cumsum(l2diff_histogram[0])
    cumulation_percentage = cumulation / np.sum(l2diff_histogram[0])
    cumulation_percentage = np.insert(cumulation_percentage, 0, 0)
    np.save(os.path.join(pred_dir, FIGURE_FILENAME + ".npy"), cumulation_percentage)

    plt.plot(bins, cumulation_percentage, color=curve_color)
    plt.title("Accuracy for {}".format(method_name))
    plt.grid()
    plt.savefig(os.path.join(pred_dir, FIGURE_FILENAME + ".png"), bbox_inches='tight', pad_inches=0)
    plt.show()


def main():
    polycnn_pred_dir = os.path.join(ROOT_DIR, "code/polycnn/pred")
    polycnn_curve_color = "#ff0000"

    unet_and_vectorization_pred_dir = os.path.join(ROOT_DIR, "code/unet_and_vectorization/pred")
    unet_and_vectorization_curve_color = "#ff8800"

    compute_stats(polycnn_pred_dir, "PolyCNN", polycnn_curve_color)
    compute_stats(unet_and_vectorization_pred_dir, "U-Net + vectorization", unet_and_vectorization_curve_color)


if __name__ == '__main__':
    main()
