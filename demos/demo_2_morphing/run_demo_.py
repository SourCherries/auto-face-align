import os
import alignfaces as af
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image as PilImage
from skimage.util import montage


# Grab faces that are already aligned from previous demo
my_project_path = os.path.dirname(os.path.abspath(__file__))
source_dir = my_project_path + os.path.sep + "an-unusual-pair" + os.path.sep

# path_parts = my_project_path.split(os.path.sep)[0:-1]
# source_dir = (os.path.sep.join(path_parts) + os.path.sep +
#              "demo_1_alignment" + os.path.sep + "faces-aligned" + os.path.sep)

file_prefix = ""
file_postfix = "jpg"

# Output folders
# morph_path = my_project_path + os.path.sep + "morphed_faces" + os.path.sep
# aperture_path = my_project_path + os.path.sep + "in_aperture" + os.path.sep

# List of available faces
# landmark_features, files = af.get_landmarks(source_dir, file_prefix=file_prefix, file_postfix=file_postfix)
# landmark_features, files = af.get_landmark_features(source_dir, output_dir="",
#                                                     exclude_features=['jawline',
#                                                     'left_iris', 'right_iris'])

af.get_landmarks(source_dir, file_prefix, file_postfix)
aligned_path = af.align_procrustes(source_dir, file_prefix, file_postfix,
                                   color_of_result="rgb")
# print("------------------------------------------------------------------")
# print("\nHere are the list of faces within\n\t" + source_dir + "\n")
# for i, f in enumerate(files[0]):
#     print(str(i) + "\t" + f + "\n")

# -----------------------------------------------------------------------------
# A continuum of 10 faces between faces A and B
do_these = [0, 1]
num_morphs = 10
# face_array, p, output_dir = af.morph_between_two_faces(source_dir, do_these, morph_path,
#                                                     num_morphs=num_morphs,
#                                            weight_texture=True)

af.get_landmarks(aligned_path, file_prefix, file_postfix)

face_array, p, morph_path = af.morph_between_two_faces(aligned_path,
                                                       do_these=do_these,
                                                       num_morphs=num_morphs,
                                                       file_prefix=file_prefix,
                                                       file_postfix=file_postfix,
                                                       new_dir = "morphed",
                                                       weight_texture=True)

# -----------------------------------------------------------------------------
# Set morphed faces within aperture

# Estimate landmarks of morphed faces.
af.get_landmarks(morph_path, "N", "png", start_fresh=True)

# Set morphed faces in an aperture.
# af.place_aperture(morph_path, aperture_path, aperture_type="MossEgg")

the_aperture, aperture_path = af.place_aperture(morph_path, "N",
                                                "png",
                                                aperture_type="MossEgg",
                                                contrast_norm="max",
                                                color_of_result="rgb")

# Plot and write montage of morphs
infile = aperture_path + "N0.png"
size = np.array(PilImage.open(infile)).shape[0:3]
all_images = np.zeros((num_morphs, size[0], size[1], size[2]))
for i in range(num_morphs):
    infile = aperture_path + "N" + str(i) + ".png"
    im = np.array(PilImage.open(infile))
    all_images[i, :, :, :] = im
im_montage = montage(all_images, multichannel=True, grid_shape=(1, 10)).astype(np.uint8)

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
fig.set_size_inches(w=2*num_morphs, h=2)
ax.imshow(im_montage, interpolation='nearest')

ax.set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
plt.margins(0,0)
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
plt.savefig("figure-demo-3.png", bbox_inches = 'tight',
            pad_inches = 0)
plt.show()
# END
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
