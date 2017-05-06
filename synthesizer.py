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

def compute_SSD(patch, patch_mask, texture_img, patch_l):
    tex_rows, tex_cols, _ = np.shape(texture_img)

    # only evaluate points that can serve as the centre point for a complete patch
    ssd_rows = tex_rows - 2 * patch_l
    ssd_cols = tex_cols - 2 * patch_l

    SSD = np.zeros((ssd_rows, ssd_cols))

    coords = np.where(patch_mask != 1)
    patch_points = patch[coords[0], coords[1]]
    patch_points = patch_points.astype('float')

    # for each possible location of the patch in the texture image
    for r in range(ssd_rows):
        for c in range(ssd_cols):
            tex_points = texture_img[(coords[0] + r), (coords[1] + c)]
            diff = patch_points - tex_points
            SSD[r, c] = np.sum(diff*diff)
    return SSD

def copy_patch(hole_img, patch_mask, texture_img,
               chosen_ctr_r, chosen_ctr_c, match_ctr_r, match_ctr_c, patch_l):
    patch_rows, patch_cols = np.shape(patch_mask)

    for r in range(patch_rows):
        for c in range(patch_cols):
            if(patch_mask[r, c] == 1):
                target_r = chosen_ctr_r - patch_l + r
                target_c = chosen_ctr_c - patch_l + c

                source_r = match_ctr_r - patch_l + r
                source_c = match_ctr_c - patch_l + c
                hole_img[target_r, target_c] = texture_img[source_r, source_c]

def find_inner_edge(hole_mask):
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

def copy_texture(hole_img, target_mask, texture_img):
    """
    Copies the given texture img to the centre of the hole in the target image. The centre
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

    img_rows, img_cols, _ = np.shape(hole_img)

    for r in range(texture_rows):
        for c in range(texture_cols):
            target_row = centre_r - tex_half_h + r
            target_col = centre_c - tex_half_w + c

            if(target_row >= 0 and target_row < img_rows 
               and target_col >= 0 and target_col < img_cols
               and target_mask[target_row, target_col] == 1):
                hole_img[target_row, target_col] = texture_img[r, c]
                target_mask[target_row, target_col] = 2

#------- The best values to use will depend a lot on the grain of the texture.
patch_l = 10 # patch length
random_SD = 2 # standard deviation for random patch selection
#-----------------------------------------------------------------------------

num_args = len(sys.argv)
if num_args != 2 and num_args != 3:
    print "usage: synthesizer.py working_img [texture_img]"
    exit()

working_file = sys.argv[1]
texture_file = sys.argv[2] if num_args == 3 else sys.argv[1]

if not os.path.isfile(working_file) or not os.path.isfile(texture_file):
    print "Image file(s) not found."
    exit()

img = Image.open(working_file).convert('RGB')
img_array = np.asarray(img, dtype=np.uint8)
img = Image.open(texture_file).convert('RGB')
texture_array = np.asarray(img, dtype=np.uint8)

if not os.path.isfile('target_region.pkl') or not os.path.isfile('texture_region.pkl'):
    print "Specify the target and texture regions first."
    exit()

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
texture_img = texture_array[min_r:max_r+1, min_c:max_c+1, :]

# hole out target region in image
target_indices = target_mask.nonzero()
hole_img = img_array.copy()
hole_img[target_indices] = 0

nrows, ncols, _ = np.shape(hole_img)

# display (for debugging purposes, remove later)
hole_display = Image.fromarray(hole_img).convert('RGB')
hole_display.show()

# cast target mask to uint8 type so we can have 3 possible elements:
# 0 means doesn't need filling, 1 means needs filling, 2 means has been filled
target_mask = target_mask.astype(np.uint8)

# copy the initial texture into the hole
copy_texture(hole_img, target_mask, texture_img)

# update pixels needing to be filled
target_indices = np.where(target_mask == 1)
to_fill = len(target_indices[0])

while to_fill > 0:
    print "Remaining pixels: ", to_fill

    # find inner edge pixels
    edge_mask = find_inner_edge(target_mask)

    edge_indices = edge_mask.nonzero()
    edge_todo = len(edge_indices[0])

    while edge_todo > 0:
        print "Edge left in this batch: ", edge_todo

        # pick a random pixel that still needs to be done
        index = np.random.randint(0, edge_todo)
        chosen_ctr_r = edge_indices[0][index]
        chosen_ctr_c = edge_indices[1][index]

        chosen_min_r = max(0, chosen_ctr_r - patch_l)
        chosen_max_r = chosen_ctr_r + patch_l
        chosen_min_c = max(0, chosen_ctr_c - patch_l)
        chosen_max_c = chosen_ctr_c + patch_l

        patch = hole_img[chosen_min_r:chosen_max_r + 1, chosen_min_c:chosen_max_c + 1, :]
        patch_mask = target_mask[chosen_min_r:chosen_max_r + 1, chosen_min_c:chosen_max_c + 1]

        ssd = compute_SSD(patch, patch_mask, texture_img, patch_l)

        # Pick random best:
        # 1. flatten and sort array
        ssd_sorted = np.sort(np.copy(ssd), axis=None)
        # 2. select random number from gaussian distribution with mean 0,
        # and get the ssd value at that index
        rand = int(round(abs(random.gauss(0, random_SD))))
        ssd_value = ssd_sorted[min(rand, np.size(ssd_sorted) - 1)]
        # 3. find which index in the original unflattened array had that value
        match_index = np.nonzero(ssd == ssd_value)

        # compute_SSD only returns values for all indices around which a patch fits within the texture
        # image bounds. Therefore a 0,0 index is really the point at patch_l, patch_l. Adjust selected
        # index to correct for this. 
        match_ctr_r = match_index[0][0] + patch_l
        match_ctr_c = match_index[1][0] + patch_l

        # pad mask to ensure that copy_patch doesn't need to worry about partial patches near image edges
        overflow_min_r = abs(min(0, chosen_ctr_r - patch_l))
        overflow_max_r = abs(min(0, nrows - (chosen_ctr_r + patch_l)))
        overflow_min_c = abs(min(0, chosen_ctr_c - patch_l))
        overflow_max_c = abs(min(0, ncols - (chosen_ctr_c + patch_l)))
        patch_mask = np.lib.pad(patch_mask, 
                                ((overflow_min_r, overflow_max_r), (overflow_min_c, overflow_max_c)), 
                                'constant', constant_values=0)

        # copy patch over
        copy_patch(hole_img, patch_mask, texture_img,
                   chosen_ctr_r, chosen_ctr_c, match_ctr_r, match_ctr_c, patch_l)
        Image.fromarray(hole_img).convert('RGB').show()

        # update pixels and number to be done
        edge_mask[chosen_min_r:chosen_max_r + 1, chosen_min_c:chosen_max_c + 1] = 0
        target_mask[chosen_min_r:chosen_max_r + 1, chosen_min_c:chosen_max_c + 1] = 2

        edge_indices = edge_mask.nonzero()
        edge_todo = len(edge_indices[0])

    # update pixels needing to be filled
    target_indices = np.where(target_mask == 1)
    to_fill = len(target_indices[0])

final_img = Image.fromarray(hole_img).convert('RGB')
final_img.show()
final_img.save('result.jpg')
