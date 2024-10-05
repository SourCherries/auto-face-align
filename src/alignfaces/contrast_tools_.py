import numpy as np
from skimage import exposure

# Ensures that values are centered on 127.5 and either reach until 0 or 255.
#   full_image, numpy image array (any range of values)
#   inner_locs, optional numpy binary map of inner face
#               if supplied, all values are normalized but only properties
#               within map are centered on 127.5 and reach until 0 or 255.
#               no clipping occurs within map, but can occur outside.
#
#   output image is numpy array, unsigned 8-bit integers.
def max_stretch_around_127(full_image, inner_locs=[]):
    if inner_locs==[]:
        inner_locs = np.ones(full_image.shape) == 1
    else:
        print("\nWarning: application of max_stretch_around_127 to subregion" +
              "can result in clipping outside that subregion.")

    inner_values = full_image[inner_locs]
    om = inner_values.mean()  # original mean value within binary map
    inner_values = inner_values - om
    if abs(inner_values.max()) > abs(inner_values.min()):
        S = 127.5 / abs(inner_values.max())
    elif abs(inner_values.max()) < abs(inner_values.min()):
        S = 127.5 / abs(inner_values.min())

    full_image = (full_image - om) * S + 127.5
    return full_image.astype(np.uint8)


# Ensures that values are centered on original mean and either reach until 0 or 255.
#   full_image, numpy image array either [0-1] or [0-255]
#               if [0-1] then multiplied by 255 to get original mean
#   inner_locs, optional numpy binary map of inner face
#               if supplied, all values are normalized but only properties
#               within map are centered on 127.5 and reach until 0 or 255.
#               no clipping occurs within map, but can occur outside.
#
#   output image is numpy array, unsigned 8-bit integers.
def max_stretch_around_original_mean(full_image, inner_locs=[]):
    if (full_image.min() >=0) and (full_image.max()<=1):
        full_image = full_image * 255
    if inner_locs==[]:
        inner_locs = np.ones(full_image.shape) == 1
    else:
        print("\nWarning: application of max_stretch_around_original_mean" +
              "to subregion can result in clipping outside that subregion.")

    inner_values = full_image[inner_locs]
    om = inner_values.mean()  # original mean value within binary map
    inner_values = inner_values - om
    if abs(inner_values.max()) > abs(inner_values.min()):
        # S = 127.5 / abs(inner_values.max())
        # [om to 255]
        # so maximum should now be equal to 255-om
        # so multiply all by S where:
        #     MX * S = 255 - om
        #     S = (255 - om) / MX
        S = (255 - om) / inner_values.max()
    elif abs(inner_values.max()) < abs(inner_values.min()):
        # S = 127.5 / abs(inner_values.min())
        # [0 to om]
        # so minimum should now be equal to -om
        # so multiply all by:
        #     MN * S = -om
        #     S = -om / MN
        S = -om / inner_values.min()
    full_image = (full_image - om) * S + om
    return full_image.astype(np.uint8)


def max_stretch(full_image, inner_locs=[]):
    if inner_locs==[]:
        inner_locs = np.ones(full_image.shape) == 1
    else:
        print("\nWarning: application of max_stretch to subregion" +
              "can result in clipping outside that subregion.")

    inner_values = full_image[inner_locs]
    omin = inner_values.min()  # original mean value within binary map
    inner_values = inner_values - omin
    omax = inner_values.max()

    full_image = (full_image - omin) / omax
    full_image = full_image * 255
    return full_image.astype(np.uint8)


def contrast_stretch(full_image, inner_locs=[], type="max"):
    if full_image.ndim == 3:
        assert (type=="max") or (type==None)
        if type == "max":
            out_image = exposure.rescale_intensity(full_image)
        return out_image
    if type == "max":
        out_image = max_stretch(full_image, inner_locs)
    elif type == "mean_127":
        out_image = max_stretch_around_127(full_image, inner_locs)
    elif type == "mean_keep":
        out_image = max_stretch_around_original_mean(full_image, inner_locs)
    elif type == None:
        out_image = full_image
    else:
        out_image = full_image
        print("Warning: Invalid argument (type) to constrast_stretch.")
    return out_image
