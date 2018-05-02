import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../utils/")
import polygon_utils


# --- Params --- #
ROOT_DIR = "../../"

IM_RES = 64

# Inputs
PRED_DIR = os.path.join(ROOT_DIR, "code/polycnn/pred")
UNET_PRED_DIR = os.path.join(ROOT_DIR, "code/unet_and_vectorization/pred")

FIGURE_FILENAME = "cumulative_accuracy"

# --- --- #

poly_cumulation_percentage = np.load(os.path.join(PRED_DIR, FIGURE_FILENAME + ".npy"))
unet_cumulation_percentage = np.load(os.path.join(UNET_PRED_DIR, FIGURE_FILENAME + ".npy"))

bins = np.arange(0, IM_RES // 2, 0.5)
plt.plot(bins, poly_cumulation_percentage, label="PolyCNN", color="#ff8800")
plt.plot(bins, unet_cumulation_percentage, label="U-Net + Douglas–Peucker ", color="#ff0000")
plt.grid()
plt.xlabel("px")
plt.ylabel("acc.")
leg = plt.legend()
plt.savefig(os.path.join("{}.compare.png".format(FIGURE_FILENAME)), bbox_inches='tight', pad_inches=0)
plt.show()
