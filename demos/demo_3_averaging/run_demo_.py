import os
import alignfaces as af
import numpy as np

# plotting results in nice figures
from skimage.util import montage
import matplotlib.pyplot as plt


def slim_fig(ax):
    ax.set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    return None


# ----------------------------------------------------------------------
# Align all faces of Angela Merkel

# Path for faces directory
merkel_folder = "faces-of-merkel"
my_project_path = os.path.dirname(os.path.abspath(__file__))
my_faces_path = my_project_path + os.path.sep + merkel_folder + os.path.sep

# Analyze all image files whose filenames have these properties ...
file_prefix = "merkel"
file_postfix = "jpg"

# Estimate landmarks.
af.get_landmarks(my_faces_path, file_prefix, file_postfix, start_fresh=True)

# Now we're ready to align the faces - via generalized Procrustes analysis.
aligned_path = af.align_procrustes(my_faces_path, file_prefix, file_postfix)

# Estimate landmarks of aligned faces.
af.get_landmarks(aligned_path, file_prefix, file_postfix)

# ----------------------------------------------------------------------
# Simple average
simple_average = af.get_mean_image(aligned_path, max_inner_face_contrast=True)


# ----------------------------------------------------------------------
# Warp each face to mean of landmarks
original_images, warped_to_mean = af.warp_to_mean_landmarks(aligned_path,
                                                            file_prefix=file_prefix,
                                                            file_postfix=file_postfix)

# Mean of warped faces
enhanced_average = warped_to_mean.mean(axis=0)

# ----------------------------------------------------------------------
# Figures to show results

# Make directory of results
# results_dir = "results"
# if not os.path.isdir(results_dir):
#     os.mkdir(results_dir)

# Display results
fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
im_montage = montage(original_images, rescale_intensity=True,
                     grid_shape=(2, 5))
ax.imshow(im_montage, cmap=plt.cm.gray, interpolation='nearest')
slim_fig(ax)
plt.savefig('A_the_many_faces_of_merkel.png', bbox_inches = 'tight', pad_inches = 0)

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
ax.imshow(simple_average, cmap=plt.cm.gray)
slim_fig(ax)
plt.savefig('B_simple_average.png', bbox_inches = 'tight', pad_inches = 0)

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
im_montage = montage(warped_to_mean, rescale_intensity=True, grid_shape=(2, 5))
ax.imshow(im_montage, cmap=plt.cm.gray, interpolation='nearest')
slim_fig(ax)
plt.savefig('C_the_many_warped_faces_of_merkel.png', bbox_inches = 'tight', pad_inches = 0)

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
ax.imshow(enhanced_average, cmap=plt.cm.gray)
slim_fig(ax)
plt.savefig('D_enhanced_average.png', bbox_inches = 'tight', pad_inches = 0)

# Compare the different average images
# from PIL import Image, ImageDraw, ImageFont
#
# im1 = Image.open('B_simple_average.png')
# im2 = Image.open('D_enhanced_average.png')
# d = ImageDraw.Draw(im1)
# d.text((10, 20), "simple", fill=(0, 0, 0))
# d = ImageDraw.Draw(im2)
# d.text((10, 20), "enhanced", fill=(0, 0, 0))
# dst = Image.new("L", (im1.width*2, im2.height))
# dst.paste(im1, (0, 0))
# dst.paste(im2, (im1.width, 0))
# dst.show()
#
# # Try different way
# im1 = np.array(im1.convert("L"))
# im2 = np.array(im2.convert("L"))

the_aperture = af.place_aperture(aligned_path, aligned_path, no_save=True)
inner_map = (the_aperture * 255) > 16

nim1 = af.contrast_stretch(simple_average, inner_locs=inner_map, type="mean_127")
nim2 = af.contrast_stretch(enhanced_average, inner_locs=inner_map, type="mean_127")

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
plt.imshow(np.c_[nim1, nim2], cmap="gray")
slim_fig(ax)
ax.text(28, 39, "simple", fontsize=36, color="k")
ax.text(280 + 28, 39, "enhanced", fontsize=36, color="k")
plt.savefig('comparison_average_types.png', bbox_inches = 'tight', pad_inches = 0)
# END
# ----------------------------------------------------------------------
