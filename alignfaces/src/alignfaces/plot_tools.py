import numpy as np
import matplotlib.pyplot as plt
import io
import json
from PIL import Image as PilImage
from matplotlib.patches import Ellipse


def plot_faces_with_landmarks(file_path, num_faces=3):
    features = ['left_eyebrow', 'left_eye', 'right_eyebrow', 'right_eye',
                'nose', 'mouth_outline', 'mouth_inner']
    with io.open(file_path + '/landmarks.txt', 'r') as f:
        imported_landmarks = json.loads(f.readlines()[0].strip())
    guys = list(imported_landmarks.keys())

    fig, ax = plt.subplots(1, num_faces)
    fig.set_size_inches(w=6*num_faces, h=10)
    for (guy, i) in zip(guys, range(num_faces)):
        gray = np.array(PilImage.open(file_path + guy).convert("L"))
        this_guy = imported_landmarks[guy]
        these_x = np.empty((0, 3))
        these_y = np.empty((0, 3))
        for f in features:
            tempy = np.array(this_guy[f])
            x = tempy[::2]
            y = tempy[1::2]
            these_x = np.append(these_x, x)
            these_y = np.append(these_y, y)
        ax[i].imshow(gray, cmap='gray', vmin=0, vmax=255)
        ax[i].plot(these_x, these_y, 'r.', linewidth=0, markersize=6)
        ax[i].axis("scaled")
        title_str = ("Close this window to continue.")
        fig.suptitle(title_str, fontsize=24)
    plt.show()
    return fig, ax


def plot_faces_with_landmarks_one_by_one(file_path):

    features = ['left_eyebrow', 'left_eye', 'right_eyebrow', 'right_eye',
                'nose', 'mouth_outline', 'mouth_inner']
    with io.open(file_path + '/landmarks.txt', 'r') as f:
        imported_landmarks = json.loads(f.readlines()[0].strip())
    guys = list(imported_landmarks.keys())

    num_faces = len(guys)

    for (guy, i) in zip(guys, range(num_faces)):
        gray = np.array(PilImage.open(file_path + guy).convert("L"))
        this_guy = imported_landmarks[guy]
        these_x = np.empty((0, 3))
        these_y = np.empty((0, 3))
        for f in features:
            tempy = np.array(this_guy[f])
            x = tempy[::2]
            y = tempy[1::2]
            these_x = np.append(these_x, x)
            these_y = np.append(these_y, y)

        wh = gray.shape[1] / gray.shape[0]
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        fig.set_size_inches(w=8*wh, h=8)

        ax.imshow(gray, cmap='gray', vmin=0, vmax=255)
        ax.plot(these_x, these_y, 'r.', linewidth=0, markersize=6)
        ax.set_xlabel(guy, fontsize=24)
        title_str = ("Face " + str(i+1) + " out of " + str(num_faces) +
                     " total." + "\n Close this window to continue.")
        plt.suptitle(title_str, fontsize=24)
        ax.axis("scaled")
        plt.show()
    return


def fit_ellipse_to_2D_cloud(data):
    """Fit ellipse to set of 2D points using PCA.

    : param data: set of 2D points
    : type data: numpy.ndarray (number of points by 2)
    """
    data_centered = data - data.mean(axis=0)
    U, s, Vt = np.linalg.svd(data_centered)
    c1 = Vt.T[:, 0]  # first eigenvector
    c2 = Vt.T[:, 1]  # second eigenvector

    T1 = np.dot(data, c1)
    T2 = np.dot(data, c2)
    transformed_data = np.c_[T1, T2]

    # Ellipse properties
    angle_deg = np.arctan2(c1[1], c1[0]) * 180 / np.pi
    cxy = tuple(data.mean(axis=0))
    major_sd, minor_sd = transformed_data.std(axis=0)
    ellipse_fit = {}
    ellipse_fit['cxy'] = cxy
    ellipse_fit['major_sd'] = major_sd
    ellipse_fit['minor_sd'] = minor_sd
    ellipse_fit['angle_deg'] = angle_deg
    return ellipse_fit


def plot_ellipse(ax, ellipse_fit, alpha=0.5, color="red"):
    """Plot ellipse fit to landmark data; diameters are 2 SD of each axis.

    : param ax: where to plot ellipse.
    : type ax: matplotlib.axes

    : param ellipse_fit: ellipse arguments output by fit_ellipse_to_2D_cloud()
    : type ellipse_fit: dict

    : param alpha: ellipse transparency
    : type alpha: float [0-1]

    : param color: color of ellipse
    : type color: str or RGB tuple
    """
    cxy = ellipse_fit['cxy']
    major_length = ellipse_fit['major_sd'] * 2
    minor_length = ellipse_fit['minor_sd'] * 2
    angle_deg = ellipse_fit['angle_deg']
    ellipse = Ellipse(cxy, major_length, minor_length, angle=angle_deg,
                      alpha=alpha, color=color)
    ax.add_artist(ellipse)
    return

# def get_mean_image(file_path, max_inner_face_contrast=False):
#     with io.open(file_path + '/landmarks.txt', 'r') as f:
#         imported_landmarks = json.loads(f.readlines()[0].strip())
#     guys = list(imported_landmarks.keys())
#     gray_0 = np.array(PilImage.open(file_path + guys[0]).convert("L"))
#     shape_0 = gray_0.shape
#
#     if max_inner_face_contrast:
#         # Normalize contrast within aperture, common mean of 127.5
#         the_aperture = place_aperture(file_path, file_path, no_save=True)
#         inner_map = (the_aperture * 255) > 16
#         gray_0 = contrast_stretch(gray_0, inner_locs=inner_map, type="mean_127")
#     else:
#         # UINT8 to double. Center and normalize
#         gray_0 = gray_0.astype(float)
#         gray_0 -= gray_0.mean()
#         gray_0 = gray_0 / gray_0.std()
#
#     mean_image = gray_0
#     for guy in guys[1:]:
#         gray = np.array(PilImage.open(file_path + guy).convert("L"))
#         if gray.shape==shape_0:
#
#             if max_inner_face_contrast:
#                 # Normalize contrast within aperture, common mean of 127.5
#                 gray = contrast_stretch(gray, inner_locs=inner_map, type="mean_127")
#             else:
#                 # UINT8 to double. Center and normalize
#                 gray = gray.astype(float)
#                 gray -= gray.mean()
#                 gray = gray / gray.std()
#
#             mean_image += gray
#         else:
#             print("_get_mean_image() requires that all images are same dimensions!!")
#             mean_image = None
#             return mean_image
#     print("Go back to [0-255]")
#
#     return mean_image


# END
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
