"""
Requires: 1) a working image and a texture image specified as arguments
2) masks defining areas within the working and texture images in pickles in the same folder
Copies the texture selection over to the undesired area of the working image and synthesizes more
of the same texture so as to fill the entire area.
"""

import random
import os.path
import pickle
import sys

from PIL import Image
import numpy as np

PATCH_L = 10 # patch length
STD_DEVIATION = 2 # standard deviation for random patch selection

def compute_ssd(patch, patch_mask, texture_img):
    """ Compute squared sum of differences for the given patch at each possible patch
    location along the texture image. """

    tex_rows, tex_cols, _ = np.shape(texture_img)

    # only evaluate points that can serve as the centre point for a complete patch
    ssd_rows = tex_rows - 2 * PATCH_L
    ssd_cols = tex_cols - 2 * PATCH_L

    ssd = np.zeros((ssd_rows, ssd_cols))

    # only consider points of interest, ie non-empty pixels
    # NTOE: currently this includes both newly added texture and pre-existing values
    # from the original image outside the hole
    coords = np.where(patch_mask != 1)
    patch_points = patch[coords[0], coords[1]]
    patch_points = patch_points.astype('float')

    # for each possible location of the patch in the texture image
    for r in range(ssd_rows):
        for c in range(ssd_cols):
            tex_points = texture_img[(coords[0] + r), (coords[1] + c)]
            diff = patch_points - tex_points
            ssd[r, c] = np.sum(diff * diff)
    return ssd

def copy_patch(target_img, patch_mask, texture_img,
               target_ctr_r, target_ctr_c, source_ctr_r, source_ctr_c):
    """ Copy patch from texture image to the chosen patch location in the target image. """

    patch_rows, patch_cols = np.shape(patch_mask)

    for r in range(patch_rows):
        for c in range(patch_cols):
            if(patch_mask[r, c] == 1):
                target_r = target_ctr_r - PATCH_L + r
                target_c = target_ctr_c - PATCH_L + c

                source_r = source_ctr_r - PATCH_L + r
                source_c = source_ctr_c - PATCH_L + c
                target_img[target_r, target_c] = texture_img[source_r, source_c]

def find_inner_edge(hole_mask):
    """ Find the edge of already transferred texture within the hole image. """

    nrows, ncols = np.shape(hole_mask)
    edge_mask = np.zeros(np.shape(hole_mask))

    for r in range(nrows):
        for c in range(ncols):
            if (hole_mask[r, c] == 1):
                if ((c > 0 and hole_mask[r, c - 1] == 2) or
                        (c < ncols - 1 and hole_mask[r, c + 1] == 2) or
                        (r > 0 and hole_mask[r - 1, c] == 2) or
                        (r < nrows - 1 and hole_mask[r + 1, c] == 2)):
                    edge_mask[r, c] = 1
    return edge_mask

def copy_texture(target_img, target_mask, texture_img):
    """
    Copy the given texture image to the centre of the hole in the target image. The centre
    here is defined as the centre of the bounding box. This approach may not work with highly
    irregular hole shapes.
    """

    target_indices = target_mask.nonzero()
    max_r = max(target_indices[0])
    min_r = min(target_indices[0])
    max_c = max(target_indices[1])
    min_c = min(target_indices[1])

    centre_r = abs(max_r - (max_r - min_r) / 2)
    centre_c = abs(max_c - (max_c - min_c) / 2)

    texture_rows, texture_cols, _ = np.shape(texture_img)
    tex_half_h = texture_rows / 2
    tex_half_w = texture_cols / 2

    img_rows, img_cols, _ = np.shape(target_img)

    for r in range(texture_rows):
        for c in range(texture_cols):
            target_row = centre_r - tex_half_h + r
            target_col = centre_c - tex_half_w + c

            if(target_row >= 0 and target_row < img_rows
               and target_col >= 0 and target_col < img_cols
               and target_mask[target_row, target_col] == 1):
                target_img[target_row, target_col] = texture_img[r, c]
                target_mask[target_row, target_col] = 2

def main():
    num_args = len(sys.argv)
    if num_args != 2 and num_args != 3:
        print "usage: synthesizer.py working_img [texture_img]"
        sys.exit(1)

    working_file = sys.argv[1]
    texture_file = sys.argv[2] if num_args == 3 else sys.argv[1]

    if not os.path.isfile(working_file) or not os.path.isfile(texture_file):
        print "Image file(s) not found."
        sys.exit(1)

    if not os.path.isfile('target_region.pkl') or not os.path.isfile('texture_region.pkl'):
        print "Specify the target and texture regions first."
        sys.exit(1)

    img = Image.open(working_file).convert('RGB')
    img_array = np.asarray(img, dtype=np.uint8)
    img = Image.open(texture_file).convert('RGB')
    texture_array = np.asarray(img, dtype=np.uint8)

    target_region_file = open('target_region.pkl', 'rb')
    target_mask = pickle.load(target_region_file)
    target_region_file.close()

    texture_region_file = open('texture_region.pkl', 'rb')
    texture_mask = pickle.load(texture_region_file)
    texture_region_file.close()

    # define texture image, adjusting selection to a rectangle
    # TODO: don't allow texture smaller than patch size
    texture_indices = texture_mask.nonzero()
    max_r = max(texture_indices[0])
    min_r = min(texture_indices[0])
    max_c = max(texture_indices[1])
    min_c = min(texture_indices[1])
    texture_img = texture_array[min_r:max_r + 1, min_c:max_c + 1, :]

    # hole out target region in image
    target_indices = target_mask.nonzero()
    target_img = img_array.copy()
    target_img[target_indices] = 0

    nrows, ncols, _ = np.shape(target_img)

    # cast target mask to uint8 type so we can have 3 possible modes:
    # 0 means doesn't need filling, 1 means needs filling, 2 means has been filled
    target_mask = target_mask.astype(np.uint8)

    # copy the initial texture into the hole
    copy_texture(target_img, target_mask, texture_img)

    # update pixels needing to be filled
    target_indices = np.where(target_mask == 1)
    total_todo = len(target_indices[0])

    while total_todo > 0:
        print "Remaining pixels: ", total_todo

        # build outwards
        edge_mask = find_inner_edge(target_mask)
        edge_indices = edge_mask.nonzero()
        edge_todo = len(edge_indices[0])

        while edge_todo > 0:

            # pick a random pixel that still needs to be done
            index = np.random.randint(0, edge_todo)
            target_ctr_r = edge_indices[0][index]
            target_ctr_c = edge_indices[1][index]

            target_min_r = max(0, target_ctr_r - PATCH_L)
            target_max_r = target_ctr_r + PATCH_L
            target_min_c = max(0, target_ctr_c - PATCH_L)
            target_max_c = target_ctr_c + PATCH_L

            patch = target_img[target_min_r:target_max_r + 1, target_min_c:target_max_c + 1, :]
            patch_mask = target_mask[target_min_r:target_max_r + 1, target_min_c:target_max_c + 1]

            ssd = compute_ssd(patch, patch_mask, texture_img)

            # Pick random best:
            # 1. flatten and sort array
            ssd_sorted = np.sort(np.copy(ssd), axis=None)
            # 2. select random number from gaussian distribution with mean 0,
            # and get the ssd value at that index
            rand = int(round(abs(random.gauss(0, STD_DEVIATION))))
            ssd_value = ssd_sorted[min(rand, np.size(ssd_sorted) - 1)]
            # 3. find which index in the original unflattened array had that value
            match_index = np.nonzero(ssd == ssd_value)

            # compute_ssd only returns values for all indices around which a patch fits
            # within the texture image bounds. Therefore a 0,0 index is really the point
            # at PATCH_L, PATCH_L. Adjust selected index to correct for this.
            source_ctr_r = match_index[0][0] + PATCH_L
            source_ctr_c = match_index[1][0] + PATCH_L

            # pad mask to ensure copy_patch doesn't need to worry about partial patches near edges
            overflow_min_r = abs(min(0, target_ctr_r - PATCH_L))
            overflow_max_r = abs(min(0, nrows - (target_ctr_r + PATCH_L)))
            overflow_min_c = abs(min(0, target_ctr_c - PATCH_L))
            overflow_max_c = abs(min(0, ncols - (target_ctr_c + PATCH_L)))
            patch_mask = np.lib.pad(
                patch_mask,
                ((overflow_min_r, overflow_max_r), (overflow_min_c, overflow_max_c)),
                'constant', constant_values=0)

            # copy patch over
            copy_patch(target_img, patch_mask, texture_img,
                       target_ctr_r, target_ctr_c, source_ctr_r, source_ctr_c)

            # update pixels and number to be done
            edge_mask[target_min_r:target_max_r + 1, target_min_c:target_max_c + 1] = 0
            target_mask[target_min_r:target_max_r + 1, target_min_c:target_max_c + 1] = 2

            edge_indices = edge_mask.nonzero()
            edge_todo = len(edge_indices[0])

        # update pixels needing to be filled
        target_indices = np.where(target_mask == 1)
        total_todo = len(target_indices[0])

    final_img = Image.fromarray(target_img).convert('RGB')
    final_img.show()
    final_img.save('result.jpg')

if __name__ == "__main__":
    main()
