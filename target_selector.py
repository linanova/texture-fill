"""
Allow a user to select a single point within an area in the given image, then
use flood fill with a color threshold to determine the edges, and save mask in a
pickle.
"""

import pickle
import sys
import os.path

from skimage import color
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

COLOR_THRESHOLD = 20

class AreaFiller(object):
    """ Allow user to define one or more areas. """

    def __init__(self, axes, img_array):
        nrows, ncols, _ = img_array.shape

        # convert rgb values to lab
        img_downscale = np.copy(img_array).astype(np.float32)/255
        self.img_values = color.rgb2lab(img_downscale)
        
        self.axes = axes
        self.target_col = None
        self.target_row = None
        self.target_color = None
        self.area_mask = np.zeros((nrows, ncols), dtype=np.bool)
        self.cid = axes.figure.canvas.mpl_connect('button_press_event', self)

    def __is_similar_color(self, row, col):
        """ Determine if the color at the given point is within the desired range. """
        candidate_color = self.img_values[row][col]
        delta_e = color.deltaE_ciede2000(candidate_color, self.target_color)
        return delta_e <= COLOR_THRESHOLD

    def __flood_fill(self):
        """
        Starting from a single point, move outwards to add all points that match
        the desired criteria.
        """
        nrows, ncols, _ = self.img_values.shape

        to_check = set()
        to_check.add((self.target_row, self.target_col))
        checked = np.zeros_like(self.area_mask)

        while to_check:
            (row, col) = to_check.pop()

            if checked[row][col]:
                continue

            checked[row, col] = True

            # if it doesn't meet the criteria move on
            if not self.__is_similar_color(row, col):
                continue

            self.area_mask[row][col] = True
            if col > 0:
                to_check.add((row, col - 1))
            if col < ncols - 1:
                to_check.add((row, col + 1))
            if row > 0:
                to_check.add((row - 1, col))
            if row < nrows - 1:
                to_check.add((row + 1, col))

        indices = np.where(self.area_mask)
        self.axes.plot(indices[1], indices[0], 'rs', linestyle='None', markersize=1)
        self.axes.figure.canvas.draw()

    def __call__(self, event):
        """ Record the selected point as the location of target color and invoke a flood fill. """
        self.target_col = int(round(event.xdata))
        self.target_row = int(round(event.ydata))
        self.target_color = self.img_values[self.target_row][self.target_col]

        self.__flood_fill()


def handle_close(event):
    """ Handle close event for plot. """
    mask = event.canvas.figure.areafiller.area_mask
    file_p = open('target_region.pkl', 'wb')
    pickle.dump(mask, file_p, -1)
    file_p.close()

def main():
    if len(sys.argv) != 2:
        print "usage: target_selector.py target_image"
        sys.exit(1)

    img_name = sys.argv[1]

    if not os.path.isfile(img_name):
        print "File not found."
        sys.exit(1)

    img = Image.open(img_name)

    img_array = np.asarray(img, dtype=np.uint8)
    nrows, ncols, _ = np.shape(img_array)
    sys.setrecursionlimit(nrows * ncols)

    fig = plt.figure()
    axes = plt.axes()

    # clear ticks
    axes.set_xticks([])
    axes.set_yticks([])

    # set size
    axes.set_ylim([0, nrows])
    axes.set_xlim([0, ncols])

    # flip y axis to get image right side up
    axes.invert_yaxis()

    axes.imshow(img_array)

    fig.areafiller = AreaFiller(axes, img_array)

    axes.set_title("Select point within area you want to replace.")
    fig.canvas.mpl_connect('close_event', handle_close)

    plt.show()

# TODOs:
# allow user to choose color threshold
# allow undo on area selection
# make the selection look better

if __name__ == "__main__":
    main()
