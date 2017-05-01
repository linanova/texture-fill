"""
Alows a user to define an area of an image and saves mask in a pickle.
"""

import os.path
import pickle
import sys

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

class PolygonBuilder(object):
    """
    Respond to user clicks by recording coordinates selected, and connecting the
    dots to provide a visualization of the region being selected. On a double click
    complete are, save pickle, and exit.
    """
    def __init__(self, axes, ncols, nrows):
        # start with empty line
        self.line, = axes.plot([0], [0])
        self.x_vals = list()
        self.y_vals = list()
        self.im_cols = ncols
        self.im_rows = nrows

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
            mask_img = Image.new('L', (self.im_cols, self.im_rows), 0)
            ImageDraw.Draw(mask_img).polygon(coords, outline=255, fill=255)
            mask_img.show()
            mask_arr = np.array(mask_img, dtype=np.uint8)

            file_p = open("texture_region.pkl", 'wb')
            pickle.dump(mask_arr, file_p, -1)
            file_p.close()
            plt.close()

def main():
    if len(sys.argv) != 2:
        print "Incorrect usage."
        exit()

    img_name = sys.argv[1]

    if not os.path.isfile(img_name):
        print "File not found."
        exit()

    # step 1: open image
    img = Image.open(img_name).convert('RGB')
    img_array = np.asarray(img, dtype=np.uint8)
    nrows, ncols, _ = img_array.shape

    # step 2: show image in plot
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

    # show image
    axes.imshow(img_array)

    # step 3: allow user to select an area within image
    fig.polygonbuilder = PolygonBuilder(axes, ncols, nrows)

    axes.set_title("Select rectangular area for texture.")
    plt.show()

# TODO: makes more sense to move the pickle saving into a close event outside the class
if __name__ == "__main__":
    main()
