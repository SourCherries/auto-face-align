# Module for doing Genearlized Procrustes Analysis (GPA).
#
# Initial is a modified and unit-tested version of:
#   https://medium.com/@olgakravchenko_34975/generalized-procrustes-analysis-with-python-numpy-c571e8e8a421
#   https://gist.github.com/olgakravchenko
import numpy as np
from math import atan, sin, cos
from scipy.linalg import norm


def _shape_array_to_matrix(shape):
    """ From [x1 y1 x2 y2 ... xp yp]
        to   [[x1 y1]
              [x2 y2]
              .
              .
              .
              [xp yp]]
    """
    assert len(shape.shape) == 1
    temp_shape = shape.reshape((-1, 2))
    return temp_shape


def _shape_matrix_to_array(shape):
    assert len(shape.shape) == 2
    assert shape.shape[1] == 2
    temp_shape = shape.reshape(-1)
    return temp_shape


def get_translation(shape):
    '''
    Calculates a translation for x and y
    axis that centers shape around the
    origin
    Args:
    shape(2n x 1 NumPy array) an array
    containing x coodrinates of shape
    points as first column and y coords
    as second column
    Returns:
    translation([x,y]) a NumPy array with
    x and y translationcoordinates
    '''
    mean_x = np.mean(shape[::2])
    mean_y = np.mean(shape[1::2])

    return np.array([mean_x, mean_y])


def translate(shape):
    '''
    Translates shape to the origin
    Args:
    shape(2n x 1 NumPy array) an array
    containing x coodrinates of shape
    points as first column and y coords
    as second column
    '''
    mean_x, mean_y = get_translation(shape)
    NS = np.copy(shape)
    NS[::2] -= mean_x
    NS[1::2] -= mean_y
    return NS


def get_rotation_matrix(theta):

    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])


def rotate(shape, theta):
    '''
    Rotates a shape by angle theta
    Assumes a shape is centered around
    origin
    Args:
    shape(2nx1 NumPy array) an shape to be rotated
    theta(float) angle in radians
    Returns:
    rotated_shape(2nx1 NumPy array) a rotated shape
    '''
    matr = get_rotation_matrix(theta)
    temp_shape = _shape_array_to_matrix(shape)
    rotated_shape = np.dot(matr, temp_shape.T).T

    return _shape_matrix_to_array(rotated_shape)


def procrustes_distance(reference_shape, shape):

    ref_x = reference_shape[::2]
    ref_y = reference_shape[1::2]

    x = shape[::2]
    y = shape[1::2]

    dist = np.sum(np.sqrt((ref_x - x)**2 + (ref_y - y)**2))

    return dist


def procrustes_analysis(reference_shape, shape):
    '''
    Scales, and rotates a shape optimally to
    be aligned with a reference shape
    Args:
        reference_shape(2nx1 NumPy array), a shape that
        serves as reference alignment

        shape(2nx1 NumPy array), a shape that is aligned

    Returns:
        aligned_shape(2nx1 NumPy array), an aligned shape
        translated to the location of reference shape
    '''
    temp_ref = translate(reference_shape)
    temp_sh = translate(shape)

    # get scale and rotation
    scale, theta = get_rotation_scale(temp_ref, temp_sh)

    # scale, rotate both shapes
    temp_sh = temp_sh / scale
    aligned_shape = rotate(temp_sh, theta)

    return aligned_shape


def generalized_procrustes_analysis(shapes, tol=0.001):
    '''
    Performs superimposition on a set of
    shapes, calculates a mean shape
    Args:
        shapes(a list of 2nx1 Numpy arrays), shapes to be aligned
            'raw values' acceptable as input; centering etc done here.
    Returns:
        mean(2nx1 NumPy array), a new mean shape
        aligned_shapes(a list of 2nx1 Numpy arrays), superimposed shapes
    '''
    # initialize Procrustes distance
    current_distance = np.inf  # ensure at least 2 iterations are done.

    # initialize a mean shape
    reference_shape = np.array(shapes[0])

    num_shapes = len(shapes)

    # create array for new shapes, add
    new_shapes = np.zeros(np.array(shapes).shape)

    D = []
    while True:

        # superimpose all shapes to current mean
        for sh in range(0, num_shapes):
            new_sh = procrustes_analysis(reference_shape, shapes[sh])
            new_shapes[sh] = new_sh

        # calculate new reference
        new_reference = new_shapes.mean(axis=0)

        new_distance = procrustes_distance(new_reference, reference_shape)
        D.append(new_distance)

        if (current_distance - new_distance) < tol:
            break

        # align the new_reference to old mean
        new_reference = procrustes_analysis(reference_shape, new_reference)

        # update mean and distance
        reference_shape = new_reference
        current_distance = new_distance

    return reference_shape, new_shapes, D


def get_rotation_scale(reference_shape, shape):
    '''
    Calculates rotation and scale
    that would optimally align shape
    with reference shape
    Args:
        reference_shape(2nx1 NumPy array), a shape that
        serves as reference for scaling and
        alignment

        shape(2nx1 NumPy array), a shape that is scaled
        and aligned

    Returns:
        scale(float), a scaling factor
        theta(float), a rotation angle in radians
    '''

    a = np.dot(shape, reference_shape) / norm(reference_shape)**2

    # separate x and y for the sake of convenience
    ref_x = reference_shape[::2]
    ref_y = reference_shape[1::2]

    x = shape[::2]
    y = shape[1::2]

    b = np.sum(x*ref_y - ref_x*y) / norm(reference_shape)**2

    scale = np.sqrt(a**2+b**2)
    theta = atan(b / max(a, 10**-10)) # avoid dividing by 0

    return scale, theta


# END
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
