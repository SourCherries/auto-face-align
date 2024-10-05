import os
import numpy as np
from PIL import Image as PilImage
from skimage.util import montage
import matplotlib.pyplot as plt


fig_height_in = 6;
img_frac_x = 0.10;
img_frac_y = 0.15;


def slim_fig(ax):
    ax.set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    return None


# databases = ["CAS-female", "CAS-male", "GUFD-female", "GUFD-male",
#              "KUFD-DC", "KUFD-ID"]
expression = ["an", "ca", "di", "fe", "ha", "ne", "sa", "sp"]
mouth = ["c", "o"]
databases = []

for mo in mouth:
    for ex in expression:
        databases.append("NIM-" + ex + "-" + mo)

all_im = []
for dbase in databases:
    in_file = "landmark-dist-" + dbase + ".png"
    this_im = np.array(PilImage.open(in_file))
    all_im.append(this_im)
h, w, c = this_im.shape

all_together = montage(all_im, grid_shape=(2, 8), multichannel=True)
wh = all_together.shape[1] / all_together.shape[0]

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
fig.set_size_inches(w=wh*fig_height_in, h=fig_height_in)
plt.imshow(all_together)
slim_fig(ax)
labels = ["a", "b", "c", "d", "e", "f", "g", "h",
          "i", "j", "k", "l", "m", "n", "o", "p"]

# px = [img_frac_x * w, img_frac_x * w + w] * 3
# py = [img_frac_y * h, img_frac_y * h, img_frac_y * h + h, img_frac_y * h + h,
#       img_frac_y * h + h*2, img_frac_y * h + h*2]

py = [img_frac_y * h] * 8 + [img_frac_y * h + h] * 8
px = [img_frac_x * w + i * w for i in range(8)] * 2

for L, X, Y in zip(labels, px, py):
    ax.text(X, Y, L, fontsize=36, color="k")
plt.savefig('landmark-dist-nim-all.png', bbox_inches = 'tight', pad_inches = 0)
plt.show()
