"""
Allows a user to select a single point within an area they want to replace, then
use flood fill with a color threshold to determine the edges of the desired area.
"""

import pickle
import sys
import os.path

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

COLOR_THRESHOLD = 20

class AreaFiller(object):
    """
    Allow user to define one or more areas.
    """

    def __init__(self, axes, img_array):
        nrows, ncols, _ = img_array.shape

        # self.selection = axes.plot([], [])
        self.axes = axes
        self.target_x = None
        self.target_y = None
        self.target_lab = None
        self.img_array = img_array
        self.area_mask = np.zeros((nrows, ncols), dtype=np.bool)
        self.cid = axes.figure.canvas.mpl_connect('button_press_event', self)


    def __meets_criteria(self, x, y):
        """
        Determine if the color at the given point is within the desired range.
        """
        r, g, b = self.img_array[y][x]
        match_rgb = sRGBColor(r, g, b, True)
        match_lab = convert_color(match_rgb, LabColor)

        delta_e = delta_e_cie2000(match_lab, self.target_lab)
        return delta_e <= COLOR_THRESHOLD

    def __flood_fill(self):
        """
        Starting from a single point, move outwards to add all points that match
        the desired criteria.
        """
        nrows, ncols, _ = self.img_array.shape

        to_check = set()
        to_check.add((self.target_x, self.target_y))

        while to_check:
            (x, y) = to_check.pop()

            # if it doesn't meet the criteria move on
            if not self.__meets_criteria(x, y):
                continue

            self.area_mask[y][x] = True
            if x > 0 and not self.area_mask[y][x-1]:
                to_check.add((x-1, y))
            if x < ncols - 1 and not self.area_mask[y][x+1]:
                to_check.add((x+1, y))
            if y > 0 and not self.area_mask[y-1][x]:
                to_check.add((x, y-1))
            if y < nrows - 1 and not self.area_mask[y+1][x]:
                to_check.add((x, y+1))

        indices = np.where(self.area_mask)
        self.axes.plot(indices[1], indices[0], 'rs', linestyle='None', markersize=1)
        self.axes.figure.canvas.draw()

    def __call__(self, event):
        """
        Record the selected point as the location to get the target color from.
        """
        self.target_x = int(round(event.xdata))
        self.target_y = int(round(event.ydata))

        # compute target color to compare to
        r, g, b = self.img_array[self.target_y][self.target_x]
        target_rgb = sRGBColor(r, g, b, True)
        self.target_lab = convert_color(target_rgb, LabColor)

        self.__flood_fill()


def handle_close(event):
    """
    Handle close event for plot.
    """
    mask = event.canvas.figure.areafiller.area_mask

    # hole out target region in image
    target_indices = mask.nonzero()
    hole_img = event.canvas.figure.areafiller.img_array.copy()
    hole_img[target_indices] = 0

    # display (can probably be removed later on, mostly for debugging purposes)
    hole_display = Image.fromarray(hole_img).convert('RGB')
    hole_display.show()

    # Save the pickle
    file_p = open('target_region.pkl', 'wb')
    pickle.dump(mask, file_p, -1)
    file_p.close()

def main():
    if len(sys.argv) != 2:
        print "Incorrect usage."
        exit()

    img_name = sys.argv[1]

    if not os.path.isfile(img_name):
        print "File not found."
        exit()

    img = Image.open(img_name)

    img_array = np.asarray(img, dtype=np.uint8)
    nrows, ncols, _ = img_array.shape
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
