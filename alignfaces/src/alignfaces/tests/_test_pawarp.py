from alignfaces2.warp_tools import pawarp
from numpy.fft  import fft2, ifft2
import numpy as np

import sys  # for exception handling
# import matplotlib.pyplot as plt


# TO DO
#   Loop through many and record.
#   Set criteria appropriately.


# -----------------------------------------------------------------------------
# Helper functions

def random_locations(im_size, num_loc=4, min_dist=5):
    """create array of num_loc points within min distance of min_dist pixels"""
    PAD = int(im_size * .15)
    locations =     (np.random.rand(1, 2) * (im_size-1-PAD*2)+PAD).astype(int)
    total = 1
    while total < num_loc:
        # Sample new point.
        loc =       (np.random.rand(1, 2) * (im_size-1-PAD*2)+PAD).astype(int)

        SAMPLE_NEW = 0;
        for SETLOC in locations:
            if np.sqrt(((SETLOC-loc)**2).sum()) <= min_dist:
                # Bad sample. Skip to beginning of loop for new sample.
                SAMPLE_NEW = 1;
        if SAMPLE_NEW:
            continue

        # Good sample. Append to results.
        locations = np.r_[locations, loc]
        total += 1
    IND = locations_to_indices(im_size, locations)
    locations = indices_to_locations(im_size, IND)
    return locations, IND


def perturb_locations(locations, max_value, change_range):
    num_targets = locations.shape[0]
    noise = np.random.rand(num_targets, 2)
    other_locations = ((noise-.5) * change_range + locations).astype(int)
    other_locations[other_locations < 0] = 0
    other_locations[other_locations > max_value] = max_value
    return other_locations


def image_with_donut_targets(im_size=64, locations=((8, 8),(32, 32))):
    """make zero image with targets at locations. also return target"""
    # Target
    B = np.zeros((3, 3))

    B[1, 1] = 9 * 8
    B[0, 1] = -9
    B[2, 1] = -9
    B[1, 0] = -9
    B[1, 2] = -9

    B[0, 0] = -9
    B[0, 2] = -9
    B[2, 0] = -9
    B[2, 2] = -9

    # Image with targets
    A = np.zeros((im_size, im_size))
    for l in locations:
        tr, tc = l
        A[1 + tr - 1, 1 + tc - 1] = B[1, 1]
        A[0 + tr - 1, 1 + tc - 1] = B[0, 1]
        A[2 + tr - 1, 1 + tc - 1] = B[2, 1]
        A[1 + tr - 1, 0 + tc - 1] = B[1, 0]
        A[1 + tr - 1, 2 + tc - 1] = B[1, 2]

        A[0 + tr - 1, 0 + tc - 1] = B[0, 0]
        A[0 + tr - 1, 2 + tc - 1] = B[0, 2]
        A[2 + tr - 1, 0 + tc - 1] = B[2, 0]
        A[2 + tr - 1, 2 + tc - 1] = B[2, 2]

    img, target = A, B
    return img, target


def fft_convolve2d(x, y):
    """ 2D convolution, using FFT"""
    pad = np.array(x.shape) - np.array(y.shape)
    if pad[0] % 2 == 0:
        rb, ra = int(pad[0]/2)+1, int(pad[0]/2)-1
    else:
        rb, ra = int(np.ceil(pad[0]/2)), int(np.floor(pad[0]/2))
    if pad[1] % 2 == 0:
        cb, ca = int(pad[1]/2)+1, int(pad[1]/2)-1
    else:
        cb, ca = int(np.ceil(pad[1]/2)), int(np.floor(pad[1]/2))
    pad_width = ((rb, ra), (cb, ca))
    py = np.pad(y, pad_width, mode="constant")

    fr = fft2(x)
    fr2 = fft2(np.flipud(np.fliplr(py)))
    m,n = fr.shape
    cc = np.real(ifft2(fr*fr2))
    cc = np.roll(cc, int(-m/2+1), axis=0)
    cc = np.roll(cc, int(-n/2+1), axis=1)
    return cc


def locations_to_indices(img_width, locations):
    num_targets = locations.shape[0]
    IND = np.zeros(num_targets, )
    for i, rc in enumerate(locations):
        r, c = rc
        this_ind = r * img_width + c
        IND[i] = this_ind
    IND.sort()
    return IND.astype(int)


def indices_to_locations(img_width, IND):
    IND.sort()
    num_targets = IND.size
    SUB = np.zeros((num_targets, 2), dtype=int)
    for i, ind in enumerate(IND):
        r = int(np.floor(ind / img_width))
        c = int(ind % img_width)
        SUB[i, :] = [r, c]
    return SUB


def top_n_locations(C, num_targets):
    cv = np.copy(C).flatten()
    si = cv.argsort()
    IND = si[-num_targets:]
    IND.sort()
    return IND


def top_n_locations_robust(C, num_targets):
    CC = np.copy(C)
    SUB = np.zeros((num_targets, 2), dtype=int)
    for i in range(num_targets):
        MC = np.where(CC==CC.max())
        r, c = MC[0][0], MC[1][0]
        SUB[i, :] = [r, c]
        # set 5 x 5 area centered on (r, c) to 0
        CC[r-1, c-1] = 0
        CC[r-1, c+0] = 0
        CC[r-1, c+1] = 0
        CC[r+0, c-1] = 0
        CC[r+0, c+0] = 0
        CC[r+0, c+1] = 0
        CC[r+1, c-1] = 0
        CC[r+1, c+0] = 0
        CC[r+1, c+1] = 0

        CC[r-2, c-2] = 0
        CC[r-2, c-1] = 0
        CC[r-2, c-0] = 0
        CC[r-2, c+1] = 0
        CC[r-2, c+2] = 0

        CC[r-1, c-2] = 0
        CC[r-1, c+2] = 0

        CC[r-0, c-2] = 0
        CC[r-0, c+2] = 0

        CC[r+1, c-2] = 0
        CC[r+1, c+2] = 0

        CC[r+2, c-2] = 0
        CC[r+2, c-1] = 0
        CC[r+2, c-0] = 0
        CC[r+2, c+1] = 0
        CC[r+2, c+2] = 0
    IND = locations_to_indices(CC.shape[1], SUB)
    SUB = indices_to_locations(CC.shape[1], IND)
    return SUB, IND


def distances_of_best_matched_points(locations, top_locations):
    # every location matched with every estimated location
    num_targets = locations.shape[0]
    O = locations
    N = top_locations
    DELTA = np.kron(O, np.ones((num_targets, 1))) - np.tile(N, (num_targets, 1))
    DIST = np.sqrt((DELTA**2).sum(axis=1))
    D = DIST.reshape((num_targets, num_targets))

    # distances between points and estimated points giving best-match
    MIN_D = np.zeros((num_targets,))
    for di in range(num_targets):
        MIN_D[di] = D.min();
        locs = np.where(D==D.min())
        R, C = locs[0][0], locs[1][0]
        D[R,:] = np.inf
        D[:,C] = np.inf
    return MIN_D


# -----------------------------------------------------------------------------
# Main

# Parameters - input image with targets centered on landmarks (locations)
im_size = 128           # length of square image, pixels
num_targets = 3         # number of targets
min_target_dist = 32    # minimum distance between landmarks, pixels

# Parameters - warped version of input image
change_range = 8        # range of random shift in landmark position, pixels
max_value=im_size-1     # maximum x and y position. minimum always 0.

# Parameters - test
iterations = 10000
ALL_D = np.zeros((iterations, num_targets))

# Perform multiple tests.
for this_it in range(iterations):
    # Create image with cross-shaped targets
    locations, target_indices = random_locations(im_size, num_targets, min_target_dist)
    img, target = image_with_donut_targets(im_size, locations)

    # Ensure that all helper functions are working properly.
    # Should be able to find exact target positions using convolution.
    C = fft_convolve2d(img, target)
    top_indices = top_n_locations(C, num_targets)
    working_functions = (top_indices==target_indices).all()
    assert working_functions, "Helper functions do not work ❌: Cannot interpret test."

    # Warp centers of targets to slightly different locations
    new_locations = perturb_locations(locations, max_value, change_range)
    wimg, tri, inpix, fwdwarpix = pawarp(img, base=new_locations, target=locations, interp='bilin')

    # Recover the original image by warping back
    wwimg, tri, inpix, fwdwarpix = pawarp(wimg, base=locations, target=new_locations, interp='bilin')

    # Look for the targets in the recovered image
    C = fft_convolve2d(wwimg, target)
    estimated_locations, top_indices = top_n_locations_robust(C, num_targets)

    # plt_1 = plt.figure(figsize=(10,10))
    # plt.imshow(wwimg, cmap="gray")
    # plt.plot(locations[:,1], locations[:,0], 'g.', markersize=24, alpha=0.5)
    # plt.plot(estimated_locations[:,1], estimated_locations[:,0], 'r+', markersize=24, alpha=0.5)
    # plt.show()

    best_distances = distances_of_best_matched_points(locations, estimated_locations)
    # print("Distances of best-matched points: ", best_distances)
    print(this_it)

    ALL_D[this_it, :] = best_distances

# Evaluate
ALL_D = ALL_D.flatten()
Q95_less_than_1_pixel = np.quantile(ALL_D, 0.95) < 1
max_offset_less_than_perturbation = ALL_D.max() < change_range
PASS = Q95_less_than_1_pixel and max_offset_less_than_perturbation
assert PASS, "Fail❌"

print("Passed 🚀")

# Passed with iterations == 10000 😊

# END
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
