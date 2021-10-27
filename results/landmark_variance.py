import alignfaces as af
import matplotlib.pyplot as plt
import numpy as np


# Helper function
def plot_landmark_dist_overlay(my_faces_path, file_prefix, file_postfix, output_file, exclude_features, elip_alpha=0.20):
    # Get average face.
    mean_face = af.get_mean_image(my_faces_path)

    # Get aperture for inner-face
    the_aperture = af.place_aperture(my_faces_path, file_prefix=file_prefix,
                                     file_postfix=file_postfix, no_save=True)

    # Maximize contrast within inner-face
    inner_map = (the_aperture * 255) > 16
    nim = af.contrast_stretch(mean_face, inner_locs=inner_map)

    # Combine face with alpha-channel for inner-face (windowed face)
    RGBA = af.make_four_channel_image(nim, the_aperture)

    # Get landmarks
    output_dir = my_faces_path  # not used (fixed later)
    landmark_features, files = af.get_landmark_features(my_faces_path, output_dir,
                                                        exclude_features=exclude_features)
    X = np.array(landmark_features['AllSubjectsLandmarksDict'])[:, 0::2]
    Y = np.array(landmark_features['AllSubjectsLandmarksDict'])[:, 1::2]
    num_landmarks = X.shape[1]

    # Plot mean face
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    fig.set_size_inches(w=6, h=10)
    ax.imshow(RGBA, cmap="gray")

    # Overlay a semi-transparent ellipse for each landmark
    cols = RGBA.shape[1]
    for i in range(num_landmarks):
        data = np.c_[X[:,i], Y[:,i]]
        ellipse_fit = af.fit_ellipse_to_2D_cloud(data)
        af.plot_ellipse(ax, ellipse_fit, alpha=elip_alpha, color="red")
        # ax.set_xlim(left=40, right=cols-41)
        ax.set_ylim(top=75, bottom=388)

    # Write figure as PNG file
    ax.set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(output_file, bbox_inches = 'tight',
                pad_inches = 0)
    plt.show()


# -----------------------------------------------------------------------------
# exc = ['jawline', 'left_iris', 'right_iris', 'mouth_inner']
#
# # GUFD male faces
# my_faces_path = "/Users/Carl/Documents/sparse-faces/white-male-aligned/"
# out_file = "landmark_dist_GUFD_males.png"
# plot_landmark_dist_overlay(my_faces_path, output_file=out_file, exclude_features=exc)
# END
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
