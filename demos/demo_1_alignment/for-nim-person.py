import os
import alignfaces as af
import matplotlib.pyplot as plt
import numpy as np

output_file = "average-of-32-identities.png"

my_project_path = os.path.dirname(os.path.abspath(__file__))
my_faces_path = my_project_path + os.path.sep + "faces-aligned" + os.path.sep
mean_face = af.get_mean_image(my_faces_path)

# Get aperture for inner-face
file_prefix = ""
file_postfix = "jpg"
the_aperture = af.place_aperture(my_faces_path, file_prefix=file_prefix,
                                 file_postfix=file_postfix, no_save=True)

# Maximize contrast within inner-face
inner_map = (the_aperture * 255) > 16
nim = af.contrast_stretch(mean_face, inner_locs=inner_map)

# Combine face with alpha-channel for inner-face (windowed face)
RGBA = af.make_four_channel_image(nim, the_aperture)

# Plot mean face
fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
fig.set_size_inches(w=6, h=10)
ax.imshow(RGBA, cmap="gray")


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
