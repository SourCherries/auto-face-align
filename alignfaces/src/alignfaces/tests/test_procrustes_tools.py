"""

CAVEATS
    Any transform requiring a rotation > abs(pi / 2) will be poor.
    The further past this threshold, the greater the error.
    Domain knowledge for faces allows us to perform a pre-processing step to
    circumvent this limitation by simply re-orienting any faces with a
    specific landmark configuration suggesting > abs(pi/2) from upright.

When I am done here,
Only change necessary is "import procrustes_tools as pt" to
"import alignfaces2.procrustes_tools as pt"

TO DO

1. make_aligned_faces module or elsewhere -- pre-processing reorient outliers.
    * because of max rotation issue
"""
#################################################################
#  IMPORT MODULES
#################################################################
# import sys
import numpy as np
# import matplotlib.pyplot as plt
# import pdb  # TEMPORARY FOR DEBUGGING
import alignfaces.procrustes_tools as pt
# import alignfaces2.procrustes_toolsRohlf as pt

#################################################################
#  TEMPORARY TO RUN WITHOUT INSTALLATION OF ALIGNFACES2.
#################################################################
# try:
#     arg = sys.argv[1]
# except IndexError:
#     raise SystemExit(f"Usage: {sys.argv[0]} <kravchenko or rohlf>")
#
# package_dir = "/Users/Carl/Studies/facepackage/alignfaces2/src/alignfaces2"
# sys.path.append(package_dir)
#
# if arg == "kravchenko":
#     import procrustes_tools as pt
# else:
#     import procrustes_toolsRohlf as pt


#################################################################
#  DEFINE SUBFUNCTIONS USED FOR UNIT TESTING
#################################################################
def matrix_to_list_of_vectors(M):
    """
    Convert npy matrix of shapes (shapes are rows) to a list of shapes.
    Entries in list are one-dimensional npy arrays.

    M is typical output of csv import.
    Output list is what's required of procrustes_tools module;
    specifically, generalized_procrustes_analysis().
    """
    M = M.tolist()
    L = [np.array(s) for s in M]
    return L


def max_abs_error_relative_to_mean_radius(standard_shape, test_shape):
    """
    Compare test_shape to standard_shape, relative to mean radius of standard.
    Returned value is a percentage.

    standard_shape  npy array
    test_shape      npy array

    Must have the same shape. Either:
        one-dimensional (single shape) or
        two-dimensional (shapes along first dimension)

    Each shape is [x1 y1 x2 y2 ... xp yp] for p points.
    """
    assert standard_shape.shape == test_shape.shape
    base_mat = standard_shape.reshape(-1, 2)
    base_radii = np.sqrt(np.dot(base_mat, base_mat.T) *
                         np.eye(base_mat.shape[0]))
    mean_radius_base = np.trace(base_radii) / base_mat.shape[0]

    max_abs_error = np.abs(standard_shape - test_shape).max()
    error_percent = max_abs_error * 100 / mean_radius_base
    return error_percent


def _affine_perturbation(base_shape, number_samples=2, kappa=10):
    # Center to 0.
    mean_x, mean_y = np.mean(base_shape[::2]), np.mean(base_shape[1::2])
    # centered = np.copy(base_shape)
    centered = base_shape.copy()
    centered[::2] -= mean_x
    centered[1::2] -= mean_y
    base_points = np.tile(centered, (number_samples, 1))

    # Random rotation & scale.
    thetas, scales = [], []
    temp_shape = centered.reshape((-1, 2)).T
    for i in range(number_samples):
        # Make affine transformation matrix.
        # theta = np.random.vonmises(mu=0, kappa=kappa)
        theta = np.random.random_sample() * (np.pi / 2)  # [0, pi/2)
        #   multiply from [-1 or 1]
        # isneg = (np.random.random() > 0.5)
        # theta = (-1 * isneg * theta) + (isneg==False) * theta  # (-pi/2, pi/2)
        scale = np.random.lognormal(mean=0.0, sigma=0.25)
        thetas.append(theta)
        scales.append(scale)
        scaled_shape = temp_shape * scale
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]])

        # Apply transform.
        rotated_shape = np.dot(rotation_matrix, scaled_shape)
        base_points[i, :] = rotated_shape.T.reshape(-1)

    # Random translation.
    shift_shapes = np.random.randint(low=0, high=10, size=(number_samples, 2))
    shift_points = np.tile(shift_shapes, (1, int(centered.size / 2)))
    shapes = base_points + shift_points

    # Format of shapes in Procrustes Tools is list of numpy arrays.
    shape_list = []
    for i in range(number_samples):
        shape_list.append(shapes[i, :])
    return {"shape_list": shape_list, "thetas": thetas,
            "scales": scales, "shift_points": shift_points,
            "meanxy": [mean_x, mean_y]}


def _affine_recovery(shapes, thetas, scales, shift_points):
    """
    Get recovered_shapes from shapes and parameters for original affine
    transformation.

    Each should be very close to base_shape
    Can then measure average distance between base_shape and each of shapes.

    But for a fair comparison with GPA output, we can also compare between the
    recovered shapes themselves.
    """
    # Set up.
    shapes_array = np.array(shapes)
    num_shapes = shapes_array.shape[0]
    num_points = int(shapes_array.shape[1] / 2)

    # Undo translation.
    shifted_shapes = shapes_array - shift_points

    # Undo rotation & scale.
    recovered_points = np.zeros(shapes_array.shape)
    for i in range(num_shapes):
        theta = -thetas[i]  # HERE
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]])

        temp_shape = shifted_shapes[i, :].reshape((-1, 2)).T
        rotated_shape = np.dot(rotation_matrix, temp_shape)
        scaled_shape = rotated_shape / scales[i]
        recovered_points[i, :] = scaled_shape.T.reshape((1, num_points * 2))
    recovered_points[:, ::2] += mean_x
    recovered_points[:, 1::2] += mean_y
    return recovered_points


def _make_shapes_around_base(base_shape, num_pairs=10):
    """
    Randomly generate pairs of shapes that are perturbations of base_shape.
    Each shape is slightly different from base_shape.
    However, each pair of shapes has a mean equal to base_shape.
    Further, all shapes are centered at origin.
    """
    assert np.isclose(base_shape[::2].mean(), 0)
    assert np.isclose(base_shape[1::2].mean(), 0)
    shapes = []
    for i in range(num_pairs):
        # Sample perturbations to first 2 points.
        # Independence of samples means that many resulting shapes will be
        #   'different' from base_shape; i.e., affine transform insufficient to
        #   match base_shape.
        first_four = np.random.normal(loc=0, scale=2, size=(4,))
        # Final point ensures that resulting shape is centered at origin.
        next_one = np.array([-first_four[0] - first_four[2]])
        last_one = np.array([-first_four[1] - first_four[3]])
        # Vector of perturbations.
        delta = np.r_[first_four, next_one, last_one]
        # First shape in this pair is a perturbation of base_shape.
        shape1 = base_shape + delta
        # Opposite perturbation to second shape in this pair.
        # Ensures mean of this pair is almost equivalent to base_shape.
        shape2 = base_shape - delta
        # Checks to ensure what was claimed is true.
        assert np.isclose(shape1[::2].mean(), 0)
        assert np.isclose(shape1[1::2].mean(), 0)
        assert np.isclose(shape2[::2].mean(), 0)
        assert np.isclose(shape2[1::2].mean(), 0)
        assert np.allclose((shape1 + shape2) / 2, base_shape)
        shapes.append(shape1)
        shapes.append(shape2)
    mean_sampled = np.array(shapes).mean(axis=0)
    assert np.allclose(mean_sampled, base_shape)
    return shapes


# def plot_shapes(shapes):
#     assert type(shapes) is list
#     assert type(shapes[0]) is np.ndarray
#     assert len(shapes[0].shape) == 1
#     shapes_npy = np.array(shapes)
#     xx = np.transpose(shapes_npy[:, ::2])
#     yy = np.transpose(shapes_npy[:, 1::2])
#     fig, axs = plt.subplots()
#     axs.plot(xx, yy)
#     axs.axis('equal')


def center_shapes_in_list(shapes):
    return [pt.translate(s) for s in shapes]


def set_unit_energy(shapes):
    if type(shapes) is list:
        new_shapes = [z / np.sqrt(np.dot(z, z)) for z in shapes]
    elif len(shapes.shape) == 2:
        new_shapes = [z / np.sqrt(np.dot(z, z)) for z in shapes]
    else:
        new_shapes = shapes / np.sqrt(np.dot(shapes, shapes))
    return new_shapes


def get_angle_vector_3p(input_m):
    m = pt._shape_array_to_matrix(input_m)
    a = np.sqrt(((m[1] - m[2])**2).sum())
    b = np.sqrt(((m[0] - m[2])**2).sum())
    c = np.sqrt(((m[0] - m[1])**2).sum())
    ra = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
    rb = np.arccos((a**2 + c**2 - b**2) / (2*a*c))
    rc = np.arccos((a**2 + b**2 - c**2) / (2*a*b))
    return np.array([ra, rb, rc]) * (180 / np.pi)


def all_comparison_triangles_isomorphic_with_standard(standard, comparison):
    standard_angles = get_angle_vector_3p(standard)
    A = [get_angle_vector_3p(P) for P in comparison]
    matches_base = [np.allclose(a, standard_angles) for a in A]
    return all(matches_base)


# Does my affine sampler distort shape?
# This code only works with triangles.
base_shape = np.array([0, 0, 15, 10, 30, 5]).astype(float)
base_angles = get_angle_vector_3p(base_shape)
num_shapes = 1000
affine = _affine_perturbation(base_shape, number_samples=num_shapes)
shapes = affine['shape_list']
assert all_comparison_triangles_isomorphic_with_standard(base_shape, shapes)


#################################################################
#  IMPORT DATA
#################################################################
# Import geomorph plethodon data & results of geomorph's gpagen() in R.
def get_geomorph_plethodon_data():
    shapes = np.loadtxt(fname='R/input_shapes.csv', delimiter=',')
    shapes = matrix_to_list_of_vectors(shapes)
    aligned_R = np.loadtxt(fname='R/align_shapes.csv', delimiter=',')
    ref_R = np.loadtxt(fname='R/align_reference.csv', delimiter=',')
    cshapes = center_shapes_in_list(shapes)
    return [cshapes, ref_R, aligned_R]

#################################################################
#  DEFINE FUNCTIONS USED FOR UNIT TESTING
#################################################################
def test_translate_1():
    """Input shape is already centered at origin. No change expected."""
    shape = np.array([-15, -5, 0, 5, 15, 0]).astype(float)
    expected_shape = shape
    new_shape = pt.translate(shape)
    assert np.allclose(new_shape, expected_shape)


def test_translate_2():
    """Result of translation known."""
    expected_shape = np.array([-15, -5, 0, 5, 15, 0]).astype(float)
    shift = np.array([2, -6, 2, -6, 2, -6]).astype(float)
    shape = expected_shape + shift
    new_shape = pt.translate(shape)
    assert np.allclose(new_shape, expected_shape)


def test_rotate_1():
    """Result of rotation known."""
    shape = np.array([0, 0, 0, 1, 1, 0]).astype(float)
    theta = np.pi / 2
    expected_shape = np.array([0, 0, -1, 0, 0, 1]).astype(float)
    new_shape = pt.rotate(shape, theta)
    assert np.allclose(new_shape, expected_shape)


def test_rotate_2():
    """Result of rotation known. Just another theta."""
    v = 1 / np.sqrt(2)
    shape = np.array([0, 0, v, v, v, -v]).astype(float)
    theta = np.pi / 4
    expected_shape = np.array([0, 0, 0, 1, 1, 0]).astype(float)
    new_shape = pt.rotate(shape, theta)
    assert np.allclose(new_shape, expected_shape)


def test_theta_in_get_rotation_scale():
    """Optimal rotation known to be pi / 2."""
    shape = np.array([0, 0, 0, 1, 1, 0]).astype(float)
    reference_shape = np.array([0, 0, -1, 0, 0, 1]).astype(float)
    scale, theta = pt.get_rotation_scale(reference_shape, shape)
    assert np.isclose(theta, np.pi / 2)


def test_scale_in_get_rotation_scale():
    """Scale of shape relative to reference known to be 1 / 2."""
    shape = np.array([0, 0, 0, 1, 1, 0]).astype(float)
    reference_shape = np.array([0, 0, -1, 0, 0, 1]).astype(float) * 2
    scale, theta = pt.get_rotation_scale(reference_shape, shape)
    assert np.isclose(scale, 1/2)


def test_procrustes_analysis():
    """
    Shape to be aligned to reference is an affine transform of reference.
    Test function for classical procrustes analysis.
    """
    reference_shape = np.array([0, 0, 0, 1, 1, 0]).astype(float) * 2
    shape = np.array([0, 0, -1, 0, 0, 1]).astype(float)
    aligned_shape = pt.procrustes_analysis(reference_shape, shape)
    reference_shape = pt.translate(reference_shape)
    aligned_shape = pt.translate(aligned_shape)
    assert np.allclose(aligned_shape, reference_shape)


def test_generalized_procrustes_analysis_distortion():
    # Does my GPA function distort shape?
    base = np.array([0, 0, 15, 10, 30, 5]).astype(float)
    affine = _affine_perturbation(base, number_samples=1000)
    shapes = affine['shape_list']

    mean_shape, new_shapes, D = pt.generalized_procrustes_analysis(shapes)
    assert all_comparison_triangles_isomorphic_with_standard(base, new_shapes)


def test_generalized_procrustes_analysis_1():
    """
    Shape to be aligned to reference is an affine transform of reference.
    Test function for generalized procrustes analysis.
    """
    shape1 = np.array([-3, -4, 3, -4, 0, 8]).astype(float)
    shape2 = np.array([-4, 3, -4, -3, 8, 0]).astype(float)
    shapes = [shape1, shape2]
    expected_reference = shape1
    reference_shape, new_shapes, D = pt.generalized_procrustes_analysis(shapes)
    reference_shape = pt.procrustes_analysis(expected_reference,
                                             reference_shape)
    assert np.allclose(reference_shape, expected_reference)


def test_generalized_procrustes_analysis_equivalent_shapes():
    # note: n should be > 3
    """
    All triangles are affine isomorphic.

    Any transform requiring a rotation > abs(pi / 2) will be poor.
    The further past this threshold, the greater the error.
    Domain knowledge for faces allows us to perform a pre-processing step to
    circumvent this limitation by simply re-orienting any faces with a
    specific landmark configuration suggesting > abs(pi/2) from upright.

    To do
        1. [here] clip sampled thetas in _affine_perturbation().
        2. [overall] pre-processing. detect outliers via features and reorient.
    """
    base_shape = np.array([0, 0, 15, 10, 30, 5]).astype(float)
    # base_shape = pt.translate(base_shape)

    num_shapes = 100
    affine = _affine_perturbation(base_shape, number_samples=num_shapes)
    shapes = affine['shape_list']
    # shapes = center_shapes_in_list(shapes)
    # pdb.set_trace()  # TEMPORARY FOR DEBUGGING
    mean_shape, new_shapes, D = pt.generalized_procrustes_analysis(shapes)

    mean_distance_among_aligned = []
    for i in range(num_shapes - 1):
        for j in range(i + 1, num_shapes):
            delta = new_shapes[i] - new_shapes[j]
            d = np.sqrt(pt._shape_array_to_matrix(delta**2).sum(axis=1))
            mean_distance_among_aligned.append(d.mean())
    assert len(mean_distance_among_aligned) == (num_shapes**2-num_shapes)/2

    base_mat = mean_shape.reshape(-1, 2)
    base_radii = np.sqrt(np.dot(base_mat, base_mat.T) *
                         np.eye(base_mat.shape[0]))
    mean_radius_base = np.trace(base_radii) / base_mat.shape[0]
    # pdb.set_trace()  # TEMPORARY FOR DEBUGGING
    error_percent = max(mean_distance_among_aligned) * 100 / mean_radius_base
    # print("Comparison of all aligned shapes generated from random affine "
    #       "transform of a base shape. Average point-wise distance for all"
    #       "pairwise comparisons of aligned shapes. Max of these relative"
    #       "to mean radius of mean aligned shape.")
    # print(f"\tMax abs error as percent of mean radius of consensus\t: {error_percent}")
    assert error_percent < 1e-10


def test_generalized_procrustes_analysis_2():
    """
    No affine transform to exactly match the 3rd shape to the rest.
    However, each shape is an isosceles triangle.
    Given that the procrustes algorithm initializes the reference shape as
    the first shape in input list of shapes, final alignments should ensure
    that the base of each triangle is horizontal and the 3rd point is centered
    at zero along the x axis.
    """
    shape1 = np.array([-3, -4, 3, -4, 0, 8]).astype(float)
    shape2 = np.array([-4, 3, -4, -3, 8, 0]).astype(float)
    shape3 = np.array([-3, -8, 3, -8, 0, 16]).astype(float)
    shapes = [shape1, shape2, shape3]
    reference_shape, new_shapes, D = pt.generalized_procrustes_analysis(shapes)

    M = np.array(new_shapes)
    first_2_pnts_horizontal = np.allclose(M[:, 1], M[:, 3])
    third_pnt_at_zero_x = np.allclose(M[:, 4], np.array([0, 0, 0]))
    assert (first_2_pnts_horizontal and third_pnt_at_zero_x)


def test_generalized_procrustes_analysis_geomorph_1():
    geo_data = get_geomorph_plethodon_data()
    shapes = geo_data[0]
    ref_R = geo_data[1]
    # Consensus shape & aligned shapes using this Python package.
    ref_python, aligned_python, D = pt.generalized_procrustes_analysis(shapes)
    # geomorph scales to unit energy, so need to rescale ref_python.
    ref_python = set_unit_energy(ref_python)
    # Compare resulting consensus shapes.
    error_percent = max_abs_error_relative_to_mean_radius(ref_R, ref_python)
    # print("Comparison of consensus shape recovered by geomorph (R) and "
    #       "facepackage (Python).")
    # print(f"\tMax abs error as percent of mean radius\t: {error_percent}")
    assert error_percent < 1


def test_generalized_procrustes_analysis_geomorph_2():
    geo_data = get_geomorph_plethodon_data()
    shapes = geo_data[0]
    aligned_R = geo_data[2]
    # Consensus shape & aligned shapes using this Python package.
    ref_python, aligned_python, D = pt.generalized_procrustes_analysis(shapes)
    # print("geomorph outputs unit scale shapes. do same with python output.")
    aligned_python = set_unit_energy(aligned_python)
    aligned_python = np.array(aligned_python)
    # pdb.set_trace()  # TEMPORARY FOR DEBUGGING
    # Compare shapes aligned by the two methods.
    error_percent = max_abs_error_relative_to_mean_radius(aligned_R,
                                                          aligned_python)
    # print("Comparison of shapes aligned by geomorph (R) and by"
    #       "facepackage (Python).")
    # print(f"\tMax abs error as percent of mean radius\t: {error_percent}")
    assert error_percent < 1


def test_generalized_procrustes_analysis_recover_donut_center():
    """
    Recover a base shape (prototype) that does not exist in analyzed shapes.
    """
    base_shape = np.array([-6, -8, 6, -8, 0, 16]).astype(float)
    shapes = _make_shapes_around_base(base_shape, num_pairs=100)
    reference_shape, new_shapes, D = pt.generalized_procrustes_analysis(shapes)

    # New reference should be the same 'shape' as the base, or very close.
    # But likely does not have the right scale and orientation.
    aligned_reference = pt.procrustes_analysis(base_shape, reference_shape)
    error_percent = max_abs_error_relative_to_mean_radius(base_shape,
                                                          aligned_reference)
    # print("Recover base shape at center of sampled shapes. "
    #       "Exact shape variable.")
    # print(f"\tMax abs error as percent of mean radius\t: {error_percent}")
    assert error_percent < 1


#################################################################
#  UNIT TESTS
#################################################################

# Basic functions.
test_translate_1()
test_translate_2()
test_rotate_1()
test_rotate_2()

# Alignment functions -
#   Shapes are equivalent except for affine transform.
#   Expect near-perfect alignments.
test_theta_in_get_rotation_scale()
test_scale_in_get_rotation_scale()
test_procrustes_analysis()
test_generalized_procrustes_analysis_1()

# *** all below involve more than 2 input shapes.
test_generalized_procrustes_analysis_distortion()
test_generalized_procrustes_analysis_equivalent_shapes()

# Alignment functions -
#   Shapes cannot be perfectly aligned by affine transform.
#       only 3 input shapes
#       only match 'shape features'
test_generalized_procrustes_analysis_2()
test_generalized_procrustes_analysis_geomorph_1()
test_generalized_procrustes_analysis_geomorph_2()
test_generalized_procrustes_analysis_recover_donut_center()


###############################################################################
###############################################################################
# Limits of simple affine.
# base_shape
# theta = np.pi / 2
# r_shape = pt.rotate(base_shape, theta)
# a_shape = pt.procrustes_analysis(base_shape, r_shape)
# plot_shapes([base_shape, a_shape])
# err = max_abs_error_relative_to_mean_radius(base_shape, a_shape)
#
# degrees = [-d for d in range(180)]
# radians = [d * np.pi/180 for d in degrees]
# E = []
# for theta in radians:
#     r_shape = pt.rotate(base_shape, theta)
#     a_shape = pt.procrustes_analysis(base_shape, r_shape)
#     err = max_abs_error_relative_to_mean_radius(base_shape, a_shape)
#     E.append(err)
# plt.plot(degrees, E)
#
# delta = np.diff(np.array(E), n=1)
# plt.plot(delta)
# delta[0:100].argmax()
# E[90]
# E[91]
# degrees[90:92]
#
# # Rotations above 90 degrees lead to jump in error.
# base_shape = np.random.standard_normal((2 * 1000, ))
# base_shape = pt.translate(base_shape)
# E = []
# for theta in radians:
#     r_shape = pt.rotate(base_shape, theta)
#     a_shape = pt.procrustes_analysis(base_shape, r_shape)
#     err = max_abs_error_relative_to_mean_radius(base_shape, a_shape)
#     E.append(err)
# plt.plot(degrees, E)
# E[90], E[91]
# degrees[90:92]
#
# # Avoid any rotation greater than pi/2 or less than -pi/2


# END
##############################################################################
##############################################################################
