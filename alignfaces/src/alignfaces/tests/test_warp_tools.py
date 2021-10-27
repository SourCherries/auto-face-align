import numpy as np
from alignfaces.warp_tools import bilin
from alignfaces.warp_tools import pawarp
from numpy.fft  import fft2, ifft2

#################################################################
#  HELPER FUNCTIONS
#################################################################
def _make_pixarr(limits):
    xmn, ymn = limits[0], limits[0]
    xmx, ymx = limits[1], limits[1]
    x = np.arange(xmn, xmx + 1)
    y = np.arange(ymn, ymx + 1)
    xx, yy = np.meshgrid(x, y)
    pixarr = (xx.flatten(), yy.flatten())
    pixarr = np.transpose(np.array(pixarr))
    return pixarr


def _random_locations(im_size, num_loc=4, min_dist=5):
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
    IND = _locations_to_indices(im_size, locations)
    locations = _indices_to_locations(im_size, IND)
    return locations, IND


def _perturb_locations(locations, max_value, change_range):
    num_targets = locations.shape[0]
    noise = np.random.rand(num_targets, 2)
    other_locations = ((noise-.5) * change_range + locations).astype(int)
    other_locations[other_locations < 0] = 0
    other_locations[other_locations > max_value] = max_value
    return other_locations


def _image_with_donut_targets(im_size=64, locations=((8, 8),(32, 32))):
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


def _fft_convolve2d(x, y):
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


def _locations_to_indices(img_width, locations):
    num_targets = locations.shape[0]
    IND = np.zeros(num_targets, )
    for i, rc in enumerate(locations):
        r, c = rc
        this_ind = r * img_width + c
        IND[i] = this_ind
    IND.sort()
    return IND.astype(int)


def _indices_to_locations(img_width, IND):
    IND.sort()
    num_targets = IND.size
    SUB = np.zeros((num_targets, 2), dtype=int)
    for i, ind in enumerate(IND):
        r = int(np.floor(ind / img_width))
        c = int(ind % img_width)
        SUB[i, :] = [r, c]
    return SUB


def _top_n_locations(C, num_targets):
    cv = np.copy(C).flatten()
    si = cv.argsort()
    IND = si[-num_targets:]
    IND.sort()
    return IND


def _top_n_locations_robust(C, num_targets):
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
    IND = _locations_to_indices(CC.shape[1], SUB)
    SUB = _indices_to_locations(CC.shape[1], IND)
    return SUB, IND


def _distances_of_best_matched_points(locations, top_locations):
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


#################################################################
#  DEFINE FUNCTIONS USED FOR UNIT TESTING
#################################################################
def test_bilin_with_integer_coordinates():
    """Simple test where warped image should be same as original."""
    im = np.array([[1, 2], [3, 4]])
    xy = (np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]))
    nRGB = 1
    RGBsub = np.array([0, 0, 0, 0])
    out = bilin(im, xy, nRGB, RGBsub)
    expected_output = im.flatten()
    assert np.allclose(out, expected_output)


def test_bilin_with_integer_coordinates_rgb():
    """Simple test where warped RGB image should be same as original."""
    im = np.array([[1, 2], [3, 4]])
    xy = (np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]))
    nRGB = 3
    RGB = np.stack((im, ) * 3, axis=-1)
    RGBsub = np.array([0, 0, 0, 0])
    for i in range(1, nRGB):
        append_this = np.ones((4,)) * i
        RGBsub = np.hstack((RGBsub, append_this))
    RGBsub = RGBsub.astype(int)
    out = bilin(RGB, xy, nRGB, RGBsub)

    pixarr = _make_pixarr([0, 1])
    pixarr_tiled = np.tile(pixarr, (3, 1))
    expected_output = RGB[pixarr_tiled[:, 1], pixarr_tiled[:, 0], RGBsub]
    assert np.allclose(out, expected_output)


def test_bilin_with_real_valued_coordinates():
    """Expected warped image is easy to calculate by hand."""
    im = np.array([[28, 24], [16, 28]])
    xy = ([0.75], [0.60])
    nRGB = 1
    RGBsub = [0]
    out = bilin(im, xy, nRGB, RGBsub)

    p = xy[0][0]
    linear1 = (1 - p) * im[0, 0] + p * im[0, 1]
    linear2 = (1 - p) * im[1, 0] + p * im[1, 1]
    p = xy[1][0]
    expected_output = (1 - p) * linear1 + p * linear2
    assert np.allclose(out[0], expected_output)


def test_pawarp():
    # Parameters - input image with targets centered on landmarks (locations)
    im_size = 128           # length of square image, pixels
    num_targets = 3         # number of targets
    min_target_dist = 32    # minimum distance between landmarks, pixels

    # Parameters - warped version of input image
    change_range = 8        # range of random shift in landmark position, pixels
    max_value=im_size-1     # maximum x and y position. minimum always 0.

    # Parameters - test
    iterations = 1000
    ALL_D = np.zeros((iterations, num_targets))

    # Perform multiple tests.
    for this_it in range(iterations):
        # Create image with cross-shaped targets
        locations, target_indices = _random_locations(im_size, num_targets, min_target_dist)
        img, target = _image_with_donut_targets(im_size, locations)

        # # Ensure that all helper functions are working properly.
        # # Should be able to find exact target positions using convolution.
        # C = fft_convolve2d(img, target)
        # top_indices = top_n_locations(C, num_targets)
        # working_functions = (top_indices==target_indices).all()
        # assert working_functions, "Helper functions do not work: Cannot interpret test."

        # Warp centers of targets to slightly different locations
        new_locations = _perturb_locations(locations, max_value, change_range)
        wimg, tri, inpix, fwdwarpix = pawarp(img, base=new_locations, target=locations, interp='bilin')

        # Recover the original image by warping back
        wwimg, tri, inpix, fwdwarpix = pawarp(wimg, base=locations, target=new_locations, interp='bilin')

        # Look for the targets in the recovered image
        C = _fft_convolve2d(wwimg, target)
        estimated_locations, top_indices = _top_n_locations_robust(C, num_targets)

        # plt_1 = plt.figure(figsize=(10,10))
        # plt.imshow(wwimg, cmap="gray")
        # plt.plot(locations[:,1], locations[:,0], 'g.', markersize=24, alpha=0.5)
        # plt.plot(estimated_locations[:,1], estimated_locations[:,0], 'r+', markersize=24, alpha=0.5)
        # plt.show()

        best_distances = _distances_of_best_matched_points(locations, estimated_locations)
        # print("Distances of best-matched points: ", best_distances)
        # print(this_it)

        ALL_D[this_it, :] = best_distances

    # Evaluate
    ALL_D = ALL_D.flatten()
    Q95_less_than_1_pixel = np.quantile(ALL_D, 0.95) < 1
    max_offset_less_than_perturbation = ALL_D.max() < change_range
    PASS = Q95_less_than_1_pixel and max_offset_less_than_perturbation
    assert PASS
#################################################################
#  UNIT TESTS
#################################################################
test_bilin_with_integer_coordinates()
test_bilin_with_integer_coordinates_rgb()
test_bilin_with_real_valued_coordinates()
test_pawarp()
