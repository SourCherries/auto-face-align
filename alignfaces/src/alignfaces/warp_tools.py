import numpy as np
from scipy.spatial import Delaunay


# bilinear interpolation
#
# input:
#   array   image, numpy array.
#   xy      image coordinates, positive real valued, tuple of arrays.
#           (length is number of pixels in array)
#   nRGB    number of color channels (third dimension of array).
#   RGBsub  channel index for each pixel in array, numpy array
#           (length is number of pixels in array)
def bilin(array, xy, nRGB, RGBsub):

    assert (array.ndim == 2) or (array.ndim == 3)
    if (array.ndim == 3):
        assert (array.shape[2] == nRGB)
    assert (len(RGBsub) == len(xy[0]) * nRGB)

    # Zero padding.
    # Handles solution to bad cases where subscripts pushed beyond image.
    if nRGB == 1:
        array = np.hstack((array, np.zeros((array.shape[0], 1))))
        array = np.vstack((array, np.zeros((1, array.shape[1]))))
    else:
        #
        # above works for single-channel grayscale.
        # now for RGB
        # print("In bilin:")
        # print(array.shape)
        # print(np.zeros((nRGB, array.shape[0], 1)).shape)
        # array = np.concatenate((array,
        #                         np.zeros((nRGB, array.shape[0], 1))), axis=2)
        # array = np.concatenate((array,
        #                         np.zeros((nRGB, 1, array.shape[1]))), axis=1)
        array = np.concatenate((array,
                                np.zeros((array.shape[0], 1, nRGB))), axis=1)
        array = np.concatenate((array,
                                np.zeros((1, array.shape[1], nRGB))), axis=0)

    xy = (np.tile(xy[0], (nRGB, 1)).flatten(),
          np.tile(xy[1], (nRGB, 1)).flatten())

    ur = (np.ceil(xy[0]).astype(int), np.floor(xy[1]).astype(int))
    ul = (np.floor(xy[0]).astype(int), np.floor(xy[1]).astype(int))
    br = (np.ceil(xy[0]).astype(int), np.ceil(xy[1]).astype(int))
    bl = (np.floor(xy[0]).astype(int), np.ceil(xy[1]).astype(int))

    bad = br[0] == ul[0]
    ur[0][bad] += 1
    br[0][bad] += 1

    bad = bl[1] == ul[1]
    br[1][bad] += 1
    bl[1][bad] += 1

    indbl = np.ravel_multi_index((bl[1], bl[0], RGBsub),
                                 (array.shape[0], array.shape[1], nRGB))
    indbr = np.ravel_multi_index((br[1], br[0], RGBsub),
                                 (array.shape[0], array.shape[1], nRGB))
    indul = np.ravel_multi_index((ul[1], ul[0], RGBsub),
                                 (array.shape[0], array.shape[1], nRGB))
    indur = np.ravel_multi_index((ur[1], ur[0], RGBsub),
                                 (array.shape[0], array.shape[1], nRGB))

    vecarray = array.flatten()

    denom = br[0] - bl[0]
    num_a = br[0] - xy[0]
    num_b = xy[0] - bl[0]
    x1out = ((num_a / denom) * vecarray[indbl] +
             (num_b / denom) * vecarray[indbr])

    x2out = ((num_a / denom) * vecarray[indul] +
             (num_b / denom) * vecarray[indur])

    denom = ul[1] - bl[1]
    num_a = ul[1] - xy[1]
    num_b = xy[1] - bl[1]
    out = (num_a / denom) * x1out + (num_b / denom) * x2out
    return out


# piecewise affine warp of target image ''im'' from target to base coords.
#
# input:
#   im      image, numpy array.
#   base    landmarks, rows are [x,y] points, numpy array.
#   target  landmarks, rows are [x,y] points, numpy array.
def pawarp(im, base, target, interp='bilin'):
    # base & target should be numpy arrays, vertices X coordinate [x/y]
    isuint8 = False
    if (im.dtype == 'uint8'):
        isuint8 = True
    assert (im.ndim == 2) or (im.ndim == 3)
    if (im.ndim == 3):
        nRGB = im.shape[2]
    else:
        nRGB = 1
    imdims = (im.shape[0], im.shape[1])
    warpim = np.zeros((im.shape[0], im.shape[1], nRGB))

    nverts = base.shape[0]
    nverts2 = target.shape[0]
    assert (nverts == nverts2)

    boxpix = np.array([[1, 1], [1, imdims[0]],
                      [imdims[1], 1], [imdims[1], imdims[0]]])
    boxpix -= 1

    pix1 = base.astype(float)
    pix2 = target.astype(float)

    pix1 = np.vstack((pix1, boxpix))
    pix2 = np.vstack((pix2, boxpix))

    # Perform Delaunay triangulation on pixel coordinates of base vertices.
    dt = Delaunay(pix1)
    tri = dt.simplices
    ntri = tri.shape[0]

    # Get the first, second, and third vertex for each triangle (x--coords).
    xio = pix1[tri[:, 0], 0]
    xi = pix2[tri[:, 0], 0]
    xjo = pix1[tri[:, 1], 0]
    xj = pix2[tri[:, 1], 0]
    xko = pix1[tri[:, 2], 0]
    xk = pix2[tri[:, 2], 0]

    # Get the first, second, and third vertex for each triangle (y--coords).
    yio = pix1[tri[:, 0], 1]
    yi = pix2[tri[:, 0], 1]
    yjo = pix1[tri[:, 1], 1]
    yj = pix2[tri[:, 1], 1]
    yko = pix1[tri[:, 2], 1]
    yk = pix2[tri[:, 2], 1]

    # Array for warp parameters (one set of params per triangle).
    # a_i for i in 1 to 6, in equation 28 of Matthews & Baker on page 145.
    wparams = np.zeros((ntri, 6))

    # Calculate warp parameters for each triangle.
    denom = (xjo - xio) * (yko - yio) - (yjo - yio) * (xko - xio)

    wparams[:, 0] = ((xio * ((xk - xi) * (yjo - yio) - (xj - xi) * (yko - yio)) +
                     yio * ((xj - xi) * (xko - xio) - (xk - xi) * (xjo - xio))) /
                     denom + xi)

    wparams[:, 3] = ((xio * ((yk - yi) * (yjo - yio) - (yj - yi) * (yko - yio)) +
                     yio * ((yj - yi) * (xko - xio) - (yk - yi) * (xjo - xio))) /
                     denom + yi)

    wparams[:, 1] = ((xj - xi) * (yko - yio) - (xk - xi) * (yjo - yio)) / denom

    wparams[:, 4] = ((yj - yi) * (yko - yio) - (yk - yi) * (yjo - yio)) / denom

    wparams[:, 2] = ((xk - xi) * (xjo - xio) - (xj - xi) * (xko - xio)) / denom

    wparams[:, 5] = ((yk - yi) * (xjo - xio) - (yj - yi) * (xko - xio)) / denom

    # Determine square bounds of pixels inside base mesh.
    xmx = int(min(np.ceil(pix1[:, 0].max()), imdims[1]))
    xmn = int(max(np.floor(pix1[:, 0].min()), 0))
    ymx = int(min(np.ceil(pix1[:, 1].max()), imdims[0]))
    ymn = int(max(np.floor(pix1[:, 1].min()), 0))

    # Array for pixel coordinates inside base mesh.
    npix = im[ymn:ymx + 1, xmn:xmx + 1].size
    pixarr = np.zeros((npix, 2))

    x = np.arange(xmn, xmx + 1)
    y = np.arange(ymn, ymx + 1)
    xx, yy = np.meshgrid(x, y)
    pixarr = (xx.flatten(), yy.flatten())
    # pixind = np.ravel_multi_index((pixarr[1], pixarr[0]), imdims)

    pixarr = np.transpose(np.array(pixarr))
    inpix = dt.find_simplex(pixarr)

    # Get only those pixels that are inside the convex hull.
    isin = np.argwhere(inpix >= 0)[:, 0]

    # Warp parameters for each pixel inside convex hull.
    wp = wparams[inpix[isin], :]

    fwdx = wp[:, 0] + wp[:, 1] * pixarr[isin, 0] + wp[:, 2] * pixarr[isin, 1]
    fwdy = wp[:, 3] + wp[:, 4] * pixarr[isin, 0] + wp[:, 5] * pixarr[isin, 1]
    if interp == 'nearest':
        fwdx = fwdx.round().astype(int)
        fwdy = fwdy.round().astype(int)

    fwdwarpix = np.transpose(np.vstack((fwdx, fwdy)))

    fwdwarpix[fwdwarpix[:, 0] < 1, 0] = 1
    fwdwarpix[fwdwarpix[:, 1] < 1, 1] = 1

    fwdwarpix[np.isnan(fwdwarpix[:, 0]), 0] = 1
    fwdwarpix[np.isnan(fwdwarpix[:, 1]), 1] = 1

    fwdwarpix[fwdwarpix[:, 0] > imdims[1], 0] = imdims[1]
    fwdwarpix[fwdwarpix[:, 1] > imdims[0], 1] = imdims[0]

    RGBsub = np.empty((1, 1))
    for RGB in range(nRGB):
        this_channel = np.ones((fwdwarpix.shape[0], 1)) * RGB
        RGBsub = np.vstack((RGBsub, this_channel))
    RGBsub = np.delete(RGBsub, (0), axis=0)
    RGBsub = RGBsub.astype(int).flatten()

    pixx = np.tile(pixarr[isin, 0], (nRGB, 1)).flatten()
    pixy = np.tile(pixarr[isin, 1], (nRGB, 1)).flatten()
    # alldims = (imdims[0], imdims[1], nRGB)
    # pixind = np.ravel_multi_index((pixy, pixx, RGBsub), alldims)
    # print("pixarr should be tiled by channel regardless of method.")
    # print("input fwdwarpix_tup grayscale for bilin.")
    # print("fwdwarpix tiled for nearest neighbor.")
    # print("pixx.shape, pixy.shape: \t")
    # print(pixx.shape)
    # print(pixy.shape)
    # print("pixarr.shape: \t")
    # print(pixarr.shape)

    # Pixarr is now expanded by channel.
    # Same as before in the case of grayscale image.
    pixarr = np.vstack((pixx, pixy)).transpose()

    if interp == 'nearest':
        # fwdwarpind = sub2ind([imdims nRGB], repmat(fwdwarpix(:,2),[nRGB 1]),
                        # repmat(fwdwarpix(:,1),[nRGB 1]), RGBsub);
        print("Nearest neighbor to do:")
        print("In Matlab, we would derive fwdwarpind.")
        print("In this case, expand pixarr & fwdwarpix across RGB channels.")
    if nRGB > 1:
        warpim = np.zeros((imdims[0], imdims[1], nRGB))
    else:
        warpim = np.zeros(imdims)
    if interp == 'nearest':
        print("fwdwarpix already reduced by isin!")
        print("To do: deal with that!")
        # expand fwdwarpix across channels
        # pixx = np.tile(pixarr[isin, 0], (nRGB, 1)).flatten()
        # pixy = np.tile(pixarr[isin, 1], (nRGB, 1)).flatten()
        # pixarr = np.vstack((pixx, pixy)).transpose()
        fwdx = np.tile(fwdwarpix[:, 0], (nRGB, 1)).flatten()
        fwdy = np.tile(fwdwarpix[:, 1], (nRGB, 1)).flatten()
        fwdwarpix = np.vstack((fwdx, fwdy)).transpose()

        print("Indexing for 3rd dimension, if there is one!")
        if nRGB > 1:
            warpim[pixarr[:, 1], pixarr[:, 0], RGBsub] = im[fwdwarpix[:, 1],
                                                            fwdwarpix[:, 0],
                                                            RGBsub]
        else:
            warpim[pixarr[:, 1], pixarr[:, 0]] = im[fwdwarpix[:, 1],
                                                    fwdwarpix[:, 0]]
    else:
        # print("now, pixarr is already reduced by isin!")
        # print("To do: deal with that!")
        #
        # print("RGBsub.shape")
        # print(RGBsub.shape)

        fwdwarpix_tup = (fwdwarpix[:, 0], fwdwarpix[:, 1])
        out = bilin(im, fwdwarpix_tup, nRGB, RGBsub)

        # print("Indexing for 3rd dimension, if there is one!")
        if nRGB > 1:
            warpim[pixarr[:, 1], pixarr[:, 0], RGBsub] = out
        else:
            warpim[pixarr[:, 1], pixarr[:, 0]] = out

    if isuint8:
        warpim = warpim.astype(np.uint8)
    return warpim, tri, inpix, fwdwarpix
# -----------------------------------------------------------------------------


# print("All above - final warp code in /warp_tests/")
