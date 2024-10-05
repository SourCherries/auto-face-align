import numpy as np

# Facial landmark detection
import dlib

# Iris localisation
# from .phase_cong_3 import phase_congruency
from skimage.morphology import dilation
from skimage.morphology import disk
from skimage import feature
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.morphology import convex_hull_image
import skimage

# Formatting landmark data
import itertools

# Active contour for pulling upper subset of jawline
# landmarks closer to hairline.
from skimage.segmentation import active_contour

# Import data file
# from pkg_resources import resource_filename
from importlib.resources import files

import os


def facial_landmarks(InputImage):

    assert InputImage.ndim == 2 or InputImage.ndim == 3, \
           'InputImage should be 2 or 3 dimensions.'

    detector = dlib.get_frontal_face_detector()

    # detect faces in the grayscale image
    rects = detector(InputImage, 0)
    if len(rects)==0:
        rects = detector(InputImage, 1)
    if len(rects)==0:
        return
    # assert len(rects)==1, 'Exactly one face must be detected in the image.'
    rect = rects[0]

    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    # predictor = dlib.shape_predictor('/Users/carl/Python Explore/models/shape_predictor_68_face_landmarks.dat')
    # predictor = dlib.shape_predictor(resource_filename('alignfaces', 'data' + os.path.sep + 'shape_predictor_68_face_landmarks.dat'))
    resource_file = files('alignfaces.data').joinpath('shape_predictor_68_face_landmarks.dat')
    predictor = dlib.shape_predictor(str(resource_file))
    shape = predictor(InputImage, rect)  # type: dlib.full_object_detection
    shape = np.array([[tp.x, tp.y] for tp in shape.parts()])  # np (68 x XY)

    assert len(shape)==68, 'Number of points returned by predictor must be 68.'

    JAWLINE_POINTS = list(range(0, 17))
    RIGHT_EYEBROW_POINTS = list(range(17, 22))
    LEFT_EYEBROW_POINTS = list(range(22, 27))
    NOSE_POINTS = list(range(27, 36))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    MOUTH_OUTLINE_POINTS = list(range(48, 61))
    MOUTH_INNER_POINTS = list(range(61, 68))
    Landmarks = {'JAWLINE_POINTS':shape[JAWLINE_POINTS,:], 'RIGHT_EYEBROW_POINTS':shape[RIGHT_EYEBROW_POINTS,:], \
    'LEFT_EYEBROW_POINTS':shape[LEFT_EYEBROW_POINTS,:], 'NOSE_POINTS':shape[NOSE_POINTS,:], \
    'RIGHT_EYE_POINTS':shape[RIGHT_EYE_POINTS,:], 'LEFT_EYE_POINTS':shape[LEFT_EYE_POINTS,:], \
    'MOUTH_OUTLINE_POINTS':shape[MOUTH_OUTLINE_POINTS,:], 'MOUTH_INNER_POINTS':shape[MOUTH_INNER_POINTS,:]}
    return Landmarks



# def Iris(InputImage,Landmarks):
#     assert InputImage.ndim==2, 'InputImage should be 2 dimensions.'
#     nrows, ncols = InputImage.shape
#     selem = disk(6) # use for dilation of binary maps
#     M, m, EO = phase_congruency(InputImage, 4, 6)
#     edges2 = feature.canny(M)
#
#     # Relevant labels for Landmarks
#     labels = dict()
#     labels[0] = 'LEFT_EYE_POINTS'
#     labels[1] = 'RIGHT_EYE_POINTS'
#     IrisPoints = dict()
#
#     hough_radii = np.arange(6, 9, 1) # range of radii to search
#
#     for fi in range(2):
#         x = Landmarks[labels[fi]][:,0]
#         y = Landmarks[labels[fi]][:,1]
#         tempbw = np.zeros((nrows,ncols))
#         tempbw[y, x] = 1
#         ch = convex_hull_image(tempbw)
#         dilated = dilation(ch, selem)
#
#         hough_res = hough_circle(dilated * edges2, hough_radii)
#         accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,total_num_peaks=8)
#
#         # Choose minimum of average pixel value within intersection of iris-disc and eye
#         total_gray = [0] * max(accums.shape)
#         i = -1
#         for center_y, center_x, radius in zip(cy, cx, radii):
#             i = i + 1
#             circy, circx = circle_perimeter(center_y, center_x, radius)
#             tempbw = np.zeros((nrows,ncols))
#             tempbw[circy, circx] = 1
#             ThisBw = convex_hull_image(tempbw)
#             total_gray[i] = np.sum(ThisBw * ch * InputImage) / np.sum(ThisBw * ch)
#
#         best_of_hough = total_gray.index(min(total_gray))
#         IrisPoints[fi] = (cy[best_of_hough], cx[best_of_hough], radii[best_of_hough])
#
#     return IrisPoints

def unpack_dict_to_vector_xy(Landmarks):
    LabelToCoordinateIndex = dict()
    CoordinateVector = []
    LabelToCoordinateIndex = dict()
    starti = 0
    for key in Landmarks.keys():
        if isinstance(Landmarks[key], np.ndarray):
            vform = Landmarks[key].flatten()
        else:
            vform = Landmarks[key]
        CoordinateVector.append(vform)
        endi = starti + len(vform)
        LabelToCoordinateIndex[key] = list(range(starti, endi))
        starti = endi
    CoordinateVector = list(itertools.chain.from_iterable(CoordinateVector))
    return CoordinateVector, LabelToCoordinateIndex

def pack_vector_xy_as_dict(CoordinateVector, LabelToCoordinateIndex):
    Landmarks = dict()
    for key in LabelToCoordinateIndex.keys():
        starti = LabelToCoordinateIndex[key][0]
        endi = LabelToCoordinateIndex[key][-1]
        clist = CoordinateVector[ starti : endi + 1 ]
        ArrayForThisKey = (np.asarray(clist)).reshape((int((endi-starti+1)/2), 2))
        Landmarks[key] = ArrayForThisKey
    return Landmarks

def pull_jawline_to_inside_of_hairline(Landmarks, Face):

    # left side
    cstart = Landmarks['JAWLINE_POINTS'][0,0] + Landmarks['JAWLINE_POINTS'][0,1]*1j
    cend   = Landmarks['JAWLINE_POINTS'][1,0] + Landmarks['JAWLINE_POINTS'][1,1]*1j
    xy = np.linspace(cstart, cend, 5)
    chunkA = np.array([np.real(xy), np.imag(xy)]).T

    cstart = Landmarks['JAWLINE_POINTS'][1,0] + Landmarks['JAWLINE_POINTS'][1,1]*1j
    cend   = Landmarks['JAWLINE_POINTS'][2,0] + Landmarks['JAWLINE_POINTS'][2,1]*1j
    xy = np.linspace(cstart, cend, 5)
    chunkB = np.array([np.real(xy), np.imag(xy)]).T

    cstart = Landmarks['JAWLINE_POINTS'][2,0] + Landmarks['JAWLINE_POINTS'][2,1]*1j
    cend   = Landmarks['JAWLINE_POINTS'][3,0] + Landmarks['JAWLINE_POINTS'][3,1]*1j
    xy = np.linspace(cstart, cend, 5)
    chunkC = np.array([np.real(xy), np.imag(xy)]).T

    upsampled_points = np.append(chunkA, chunkB[1:,:], axis=0)
    upsampled_points = np.append(upsampled_points, chunkC[1:,:], axis=0)

    init = upsampled_points

    if (int((skimage.__version__).split(".")[1]) > 19):
        # switch columns of init
        init = np.roll(init, 1, axis=1)
        snake = active_contour(Face, init, boundary_condition='free-fixed',
                               alpha=0.1, beta=1.0, w_line=0, w_edge=5,
                               gamma=0.1, convergence=0.01)
        # switch columns of snake
        snake = np.roll(snake, 1, axis=1)
    elif (int((skimage.__version__).split(".")[1]) > 15):
        # switch columns of init
        init = np.roll(init, 1, axis=1)
        snake = active_contour(Face, init, boundary_condition='free-fixed',
                               alpha=0.1, beta=1.0, w_line=0, w_edge=5,
                               gamma=0.1, convergence=0.01, coordinates='rc')
        # switch columns of snake
        snake = np.roll(snake, 1, axis=1)
    else:
        snake = active_contour(Face, init, bc='free-fixed',
                               alpha=0.1, beta=1.0, w_line=0, w_edge=5,
                               gamma=0.1, convergence=0.01)

    Landmarks['JAWLINE_POINTS'][0:3,:] = snake[0:12:4,:]

    # right side
    cstart = Landmarks['JAWLINE_POINTS'][-1,0] + Landmarks['JAWLINE_POINTS'][-1,1]*1j
    cend   = Landmarks['JAWLINE_POINTS'][-2,0] + Landmarks['JAWLINE_POINTS'][-2,1]*1j
    xy = np.linspace(cstart, cend, 5)
    chunkA = np.array([np.real(xy), np.imag(xy)]).T

    cstart = Landmarks['JAWLINE_POINTS'][-2,0] + Landmarks['JAWLINE_POINTS'][-2,1]*1j
    cend   = Landmarks['JAWLINE_POINTS'][-3,0] + Landmarks['JAWLINE_POINTS'][-3,1]*1j
    xy = np.linspace(cstart, cend, 5)
    chunkB = np.array([np.real(xy), np.imag(xy)]).T

    cstart = Landmarks['JAWLINE_POINTS'][-3,0] + Landmarks['JAWLINE_POINTS'][-3,1]*1j
    cend   = Landmarks['JAWLINE_POINTS'][-4,0] + Landmarks['JAWLINE_POINTS'][-4,1]*1j
    xy = np.linspace(cstart, cend, 5)
    chunkC = np.array([np.real(xy), np.imag(xy)]).T

    upsampled_points = np.append(chunkA, chunkB[1:, :], axis=0)
    upsampled_points = np.append(upsampled_points, chunkC[1:, :], axis=0)

    init = upsampled_points

    if (int((skimage.__version__).split(".")[1]) > 19):
        # switch columns of init
        init = np.roll(init, 1, axis=1)
        snake = active_contour(Face, init, boundary_condition='free-fixed',
                               alpha=0.1, beta=1.0, w_line=0, w_edge=5,
                               gamma=0.1, convergence=0.01)
        # switch columns of snake
        snake = np.roll(snake, 1, axis=1)
    elif (int((skimage.__version__).split(".")[1]) > 15):
        # switch columns of init
        init = np.roll(init, 1, axis=1)
        snake = active_contour(Face, init, boundary_condition='free-fixed',
                               alpha=0.1, beta=1.0, w_line=0, w_edge=5,
                               gamma=0.1, convergence=0.01, coordinates='rc')
        # switch columns of snake
        snake = np.roll(snake, 1, axis=1)
    else:
        snake = active_contour(Face, init, bc='free-fixed', alpha=0.1,
                               beta=1.0, w_line=0, w_edge=5, gamma=0.1,
                               convergence=0.01)

    Landmarks['JAWLINE_POINTS'][-1:-4:-1, :] = snake[0:12:4, :]
    return Landmarks
###############################################################################
