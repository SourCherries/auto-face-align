import os
import numpy as np
from PIL import Image as PilImage
from skimage.util import montage
import matplotlib.pyplot as plt

fnames = [["A_the_many_faces_of_merkel.png", "B_simple_average.png"],
          ["C_the_many_warped_faces_of_merkel.png", "D_enhanced_average.png"]]

imgs = []
for r in range(2):
    offspring = []
    for c in range(2):
        this_file = fnames[r][c]
        this_im = np.array(PilImage.open(this_file))
        offspring.append(this_im)
    imgs.append(offspring)

fig, axs = plt.subplots(ncols=3, nrows=2)
gs1 = axs[0, 0].get_gridspec()
gs2 = axs[1, 0].get_gridspec()

for ax in axs[0, 0:2]:
    ax.remove()
axbig1 = fig.add_subplot(gs1[0, 0:2])
axbig1.imshow(imgs[0][0])
axs[0][2].imshow(imgs[0][1])

for ax in axs[1, 0:2]:
    ax.remove()
axbig2 = fig.add_subplot(gs2[1, 0:2])
axbig2.imshow(imgs[1][0])
axs[1][2].imshow(imgs[1][1])

axbig1.set_axis_off()
axbig1.xaxis.set_major_locator(plt.NullLocator())
axbig1.yaxis.set_major_locator(plt.NullLocator())

axbig2.set_axis_off()
axbig2.xaxis.set_major_locator(plt.NullLocator())
axbig2.yaxis.set_major_locator(plt.NullLocator())

axs[0][2].set_axis_off()
axs[0][2].xaxis.set_major_locator(plt.NullLocator())
axs[0][2].yaxis.set_major_locator(plt.NullLocator())

axs[1][2].set_axis_off()
axs[1][2].xaxis.set_major_locator(plt.NullLocator())
axs[1][2].yaxis.set_major_locator(plt.NullLocator())

plt.gcf().text(0.01, 0.90, "a", fontsize=24)
plt.gcf().text(0.65, 0.90, "b", fontsize=24)

plt.gcf().text(0.01, 0.43, "c", fontsize=24)
plt.gcf().text(0.65, 0.43, "d", fontsize=24)

fig.tight_layout(pad=(2))

plt.savefig('figure-enhanced-average.png', bbox_inches = 'tight', pad_inches = 0)

plt.show()



# END
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
