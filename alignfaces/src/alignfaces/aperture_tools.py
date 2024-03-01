import numpy as np
# Might need to import below into make_aligned_faces:
#   from cv2 import imread, cvtColor, COLOR_BGR2GRAY, imwrite
from skimage.filters import gaussian


# Function to fit an ellipse using a very simple method.
#   Semi-major axis (vertical) length is fixed as argument.
#   Increase length of semi-minor until widest distance among landmarks fits.
#   Immediate code above is redundant with this function.
def fit_ellipse_semi_minor(semi_major, landmarks, center):
    X, Y = landmarks[0], landmarks[1]
    CX, CY = center[0], center[1]
    Xc = X - CX
    Yc = Y - CY
    a_min = np.floor((Xc.max() - Xc.min()) * 3 / 10)
    a = a_min
    all_in = (((Xc**2/a**2) + (Yc**2/semi_major**2)) <= 1).all()
    while (not all_in):
        a += 1
        all_in = (((Xc**2/a**2) + (Yc**2/semi_major**2)) <= 1).all()
    return a


def make_ellipse_map(semi_minor, semi_major, center, size, soften=True):
    CX, CY = center[0], center[1]
    x = np.array([i-CX for i in range(size[1])])
    y = np.array([i-CY for i in range(size[0])])
    xv, yv = np.meshgrid(x, y)
    R = (xv**2) / semi_minor**2 + (yv**2) / semi_major**2
    if soften:
        # Soften edges using Butterworth as a function of radius from (CX, CY)
        filter_n = 10
        aperture = 1 / np.sqrt(1 + R**(2*filter_n))
    else:
        aperture = R <= 1
    return aperture


# Function to make a binary map of a circle within image of size = size.
def make_circle_map(cxy, radius, size):
    size = (size[1], size[0])
    xx = np.array([[x - cxy[0] for x in range(1, size[0]+1)]
                   for y in range(size[1])])
    yy = np.array([[y - cxy[1] for y in range(1, size[1]+1)]
                   for x in range(size[0])]).T
    rr = np.sqrt(xx**2 + yy**2)
    return rr <= radius


# Function to make binary map selecting for entire image area below below_y
def make_map_below_y(below_y, size):
    size = (size[1], size[0])
    yy = np.array([[y for y in range(1, size[1]+1)] for x in range(size[0])]).T
    return yy > below_y


# Function to make a binary aperture in shape of Moss's Egg.
#
#   1. ABC isosceles with point B facing down
#       a. define upc, midpoint between A and C
#       b. upc fraction along vector from mean of inter-eye midpoints
#           to center of all landmarks
#           i. fraction default is 1/4 but set as argument
#       c. radius_upper is fraction of ellipse_width
#           i. defined in ellipse-fitting functions
#           ii. fraction default is 47/100 but set as argument
#       d. A is upc shifted left by radius_upper
#       e. C is upc shifted right by radius_upper
#       f. B[x] is upc[x] and B[y] is mean of all nose-tips
#   2. Rest of procedure follows basic construction of Moss's egg
def make_moss_egg(landmark_features, center, size,
                  fraction_width=47/100, soften=True):
    CX, CY = center[0], center[1]

    # Set radius_upper using same method used when fitting an elliptical
    # aperture.
    shapes = np.array(landmark_features['AllSubjectsLandmarksDict'])
    X = shapes[:, 0::2].reshape(-1,)
    Y = shapes[:, 1::2].reshape(-1,)
    # Longest vertical length of ellipse that fits within image.
    if (size[0] / 2) < CY:
        ellipse_height = (size[0] - CY) * 2
    elif (size[0] / 2) > CY:
        ellipse_height = CY * 2
    else:
        ellipse_height = size[0]
    semi_major = ellipse_height / 2
    semi_minor = fit_ellipse_semi_minor(semi_major=semi_major,
                                        landmarks=(X, Y),
                                        center=(CX, CY))
    ellipse_width = semi_minor * 2
    radius_upper = ellipse_width * fraction_width

    # Upper circle, centered on upc (midpoint of AC in ABC).
    #   Top half defines top of Moss Egg.
    to_center = 1 / 4
    eye_midpoints = landmark_features['eye_midpoints']
    eye_midpoint = np.array(eye_midpoints).mean(axis=0)
    upc = ((CX, CY) - eye_midpoint) * to_center + (eye_midpoint)
    horizontal_alignment = upc[0]

    # Now make two large circles whose intersection defines middle part.

    # Large circle on left, centered on cac
    radius_large = radius_upper * 2
    cac = (horizontal_alignment - radius_upper, upc[1])

    # Large circle on right, centered on cbc
    cbc = (horizontal_alignment + radius_upper, upc[1])

    # Now make small circle at bottom, centered on lm.
    nosey = np.array(landmark_features['nose_tips']).mean(axis=0)[1]
    lm = (horizontal_alignment, nosey)

    # Isosceles triangle cac -- lm -- cbc (ABC) with apex at lm.
    # Ensure that angle at lm is greater than 60 degrees.
    v1 = np.asarray(cac) - np.asarray(lm)
    v2 = np.asarray(cbc) - np.asarray(lm)
    acos = np.sum(v1 * v2) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))
    DegABC = np.arccos(acos) * 180 / np.pi
    assert DegABC > 60

    # Line defined by A (center of large circle) to lm.
    #   m * x + y_intercept
    #   m * x + c
    delta = np.array(lm) - cac
    m = delta[1] / delta[0]
    t_intercept = -cac[0] / delta[0]
    y_intercept = t_intercept * delta[1] + cac[1]

    # Intersection of Ca with above line.
    #
    # (x - cac[0])**2 + (m * x + y_intercept - cac[1])**2 = radius_large**2
    # (x -      p)**2 + (m * x +           c -      q)**2 =            r**2
    A = m**2 + 1
    B = 2 * (m * y_intercept - m*cac[1] - cac[0])
    C = (cac[1]**2 - radius_large**2 + cac[0]**2 -
         2*y_intercept*cac[1] + y_intercept**2)

    assert B**2 - 4*A*C > 0

    # Radius defined by distance from lm to above intersection.
    # x_m = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
    x_p = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
    Ex = x_p
    Ey = m * Ex + y_intercept
    lower_radius = np.sqrt((((Ex, Ey) - np.array(lm))**2).sum())

    Ca = make_circle_map(cxy=cac, radius=radius_large, size=size)
    Cb = make_circle_map(cxy=cbc, radius=radius_large, size=size)
    Cu = make_circle_map(cxy=upc, radius=radius_upper, size=size)
    Cc = make_circle_map(cxy=lm, radius=lower_radius, size=size)

    # LM1 = make_map_below_y(below_y=horizontal_alignment, size=size)
    LM1 = make_map_below_y(below_y=upc[1], size=size)
    LM2 = make_map_below_y(below_y=Ey, size=size)

    EggA = Cu
    EggB = Ca & Cb & LM1 & (LM2 == False)
    EggC = Cc & LM2
    # plt.imshow(np.c_[EggA, EggB, EggC])

    MossEgg = EggA | EggB | EggC

    if soften:
        ME = MossEgg.astype(float)
        IP = landmark_features['IrisPoints']
        IPD = [np.sqrt(sum((I[1] - I[0])**2)) for I in IP]
        sigma = round(np.asarray(IPD).mean() * 0.05)
        MossEgg = gaussian(ME, sigma=(sigma, sigma),
                           truncate=3.5 * sigma)
        # MossEgg = gaussian(ME, sigma=(sigma, sigma),
        #                    truncate=3.5 * sigma, multichannel=False)

    # package critical variables for visualizing moss's egg construction
    egg_params = {}
    egg_params['A'] = cac
    egg_params['B'] = lm
    egg_params['C'] = cbc
    egg_params['upc'] = upc
    egg_params['radius_large'] = radius_large
    egg_params['radius_upper'] = radius_upper
    egg_params['radius_lower'] = lower_radius

    return MossEgg, egg_params


# Pack all into a four-channel image of unsigned 8-bit integers.
def make_four_channel_image(img, aperture):
    # assert aperture.min() >= 0 and aperture.max() <= 1
    if not (aperture.min() >= 0 and aperture.max() <= 1):
        aperture = aperture - aperture.min()
        aperture = aperture / aperture.max()
    alpha = (aperture * 255).astype(np.uint8)
    if img.ndim == 2:
        assert type(img[0, 0]) is np.uint8
        size = img.shape
        BGRA = np.zeros((size[0], size[1], 4), np.uint8)
        for i in range(3):
            BGRA[:, :, i] = img
        BGRA[:, :, 3] = alpha
    elif img.ndim == 3:
        assert type(img[0, 0, 0]) is np.uint8
        size = img.shape
        BGRA = np.zeros((size[0], size[1], 4), np.uint8)
        for i in range(3):
            BGRA[:, :, i] = img[:, :, i]
        BGRA[:, :, 3] = alpha
    else:
        BGRA = []
        print("Warning: Image is neither grayscale nor RGB.")
    return BGRA
