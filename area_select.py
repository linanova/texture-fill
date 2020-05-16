""" Allow a user to select an area in the given image and save mask in a pickle.

Two types of selection are possible:
1. Flood fill selection - user selects a single point and the edges of the area
are determined by color theshold comparison.
2. Polygon selection - user selects multiple points which are treated as vertices
for the desired area.
"""

from argparse import ArgumentParser
import pickle
import sys

from matplotlib import pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image, ImageDraw
from skimage import color

import utils

class PolygonBuilder():
    """
    Respond to user clicks by recording coordinates and connecting the dots to provide a
    visualization of the region being selected. On a double click complete area by connecting
    back to the beginning.
    """
    def __init__(self, axes, ncols, nrows):
        # start with empty line
        self.line, = axes.plot([0], [0], '--w', linewidth=2,
                               path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])
        self.x_vals = list()
        self.y_vals = list()
        self.img_cols = ncols
        self.img_rows = nrows
        self.area_mask = None
        self.cid = self.line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.xdata is None:
            return

        self.x_vals.append(int(event.xdata))
        self.y_vals.append(int(event.ydata))

        self.line.set_data(self.x_vals, self.y_vals)
        self.line.figure.canvas.draw()

        if event.dblclick:
            # build list of coordinates in form (x1,y1,x2,y2,x3...)
            coords = list()
            for i in range(len(self.x_vals)):
                coords.append(self.x_vals[i])
                coords.append(self.y_vals[i])

            # create mask image defining the texture area
            mask_img = Image.new('L', (self.img_cols, self.img_rows), 0)
            ImageDraw.Draw(mask_img).polygon(coords, outline=255, fill=255)
            self.area_mask = np.array(mask_img, dtype=np.uint8)

            # connect to starting point and show completed selection
            self.x_vals.append(self.x_vals[0])
            self.y_vals.append(self.y_vals[0])
            self.line.set_data(self.x_vals, self.y_vals)
            self.line.figure.canvas.draw()

            # disconnect event handler to avoid undesired effects on further clicks
            self.line.figure.canvas.mpl_disconnect(self.cid)

class FloodFiller():
    """
    Respond to user clicks by performing a flood fill outward from the selected
    points and provide visualization of current selection.
    """
    COLOR_THRESHOLD = 15

    def __init__(self, axes, img_array):
        # convert rgb values to lab
        img_downscale = np.copy(img_array).astype(np.float32)/255
        self.img_values = color.rgb2lab(img_downscale)

        self.axes = axes
        self.target_col = None
        self.target_row = None
        self.target_color = None
        self.area_mask = None
        self.cid = axes.figure.canvas.mpl_connect('button_press_event', self)

    def __is_similar_color(self, row, col):
        """Determine if the color at the given point is within the desired range."""
        candidate_color = self.img_values[row][col]
        delta_e = color.deltaE_ciede94(candidate_color, self.target_color)
        return delta_e <= self.COLOR_THRESHOLD

    def __flood_fill(self):
        """
        Starting from a single point, move outwards to add all points that match
        the desired criteria.
        """
        nrows, ncols, _ = self.img_values.shape
        if self.area_mask is None:
            self.area_mask = np.zeros((nrows, ncols), dtype=np.bool)

        to_check = set()
        to_check.add((self.target_row, self.target_col))
        checked = np.zeros_like(self.area_mask)

        while to_check:
            (row, col) = to_check.pop()

            # if we've already seen it, move on
            # also check mask in case point was added in a prevoius selection
            if checked[row][col] or self.area_mask[row][col]:
                continue

            checked[row, col] = True

            # if it doesn't meet the criteria, move on
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
        self.axes.plot(indices[1], indices[0], '.', color='0.5', linestyle='None', markersize=2)
        self.axes.figure.canvas.draw()

    def __call__(self, event):
        """Record the target color and invoke a flood fill."""
        if event.xdata is None:
            return

        self.target_col = int(round(event.xdata))
        self.target_row = int(round(event.ydata))
        self.target_color = self.img_values[self.target_row][self.target_col]

        self.__flood_fill()


def handle_close(event, fname):
    """Handle close event for plot by saving the selection if one was made."""
    mask = event.canvas.figure.selector.area_mask

    if mask is not None:
        file_p = open(fname, 'wb')
        pickle.dump(mask, file_p, -1)
        file_p.close()

def main():
    """
    Open and display image. Allow user to choose between making a target or texture
    selection and initialize appropriate selector.
    """

    parser = ArgumentParser()
    parser.add_argument(
        "image", help="the image to use for area selection",
        type=lambda arg: utils.parse_image_argument(parser, arg))
    args = parser.parse_args()

    img = args.image

    img_array = np.asarray(img, dtype=np.uint8)
    nrows, ncols, _ = np.shape(img_array)
    sys.setrecursionlimit(nrows * ncols)

    # remove the standard pyplot toolbar
    plt.rcParams['toolbar'] = 'None'

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

    print("Would you like to select the region to be filled (0) or the sample texture region (1)?")

    valid_input = False
    while not valid_input:
        answer = input("0 or 1: ")
        if answer == "0" or answer == "1":
            valid_input = True

    if answer == "0":
        fname = 'target_region.pkl'
        fig.selector = FloodFiller(axes, img_array)
        axes.set_title("Select point within area you want to replace.")
    else:
        fname = 'texture_region.pkl'
        fig.selector = PolygonBuilder(axes, ncols, nrows)
        axes.set_title("Click to define an area of texture (double click to end selection).")


    fig.canvas.mpl_connect('close_event', lambda event: handle_close(event, fname))
    plt.show()

if __name__ == "__main__":
    main()
