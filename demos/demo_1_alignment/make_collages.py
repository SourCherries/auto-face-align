import os
import shutil
import random

from collage_maker import make_collage

from PIL import Image

# ------------------------------------------------------------------------
# Helper function
def getAllFiles(dirName):
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    return listOfFiles


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Collage of original faces

# ------------------------------------------------------------------------
# Temporarily copy images from all subfolders into a single folder
temp_folder = "all-together"
os.mkdir(temp_folder)

for sub_folder in ["faces/premium", "faces/popular"]:
    listOfFiles = getAllFiles(sub_folder)
    images_A = [fn for fn in listOfFiles if os.path.splitext(fn)[1].lower() in ('.jpg', '.jpeg', '.png')]
    for fn in images_A:
        shutil.copy(fn, temp_folder)


# ------------------------------------------------------------------------
# Create collage
output = "collage_originals.png"
width = 1024*2  #800
init_height = 256  #250

files = [os.path.join(temp_folder, fn) for fn in os.listdir(temp_folder)]
images = [fn for fn in files if os.path.splitext(fn)[1].lower() in ('.jpg', '.jpeg', '.png')]

random.shuffle(images)
make_collage(images, output, width, init_height)


# ------------------------------------------------------------------------
# Remove temporary folder
for fn in images:
    os.remove(fn)

os.rmdir(temp_folder)



# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Montage of aligned and apertured images

# ------------------------------------------------------------------------
# Temporarily copy images from all subfolders into a single folder
temp_folder = "all-together"
os.mkdir(temp_folder)

for sub_folder in ["faces-aligned-windowed/premium", "faces-aligned-windowed/popular"]:
    listOfFiles = getAllFiles(sub_folder)
    images_A = [fn for fn in listOfFiles if os.path.splitext(fn)[1].lower() in ('.jpg', '.jpeg', '.png')]
    for fn in images_A:
        shutil.copy(fn, temp_folder)


# ------------------------------------------------------------------------
# Create collage
files = [os.path.join(temp_folder, fn) for fn in os.listdir(temp_folder)]
images = [fn for fn in files if os.path.splitext(fn)[1].lower() in ('.jpg', '.jpeg', '.png')]

images_point = [Image.open(x) for x in images]
widths, heights = zip(*(i.size for i in images_point))

total_width = sum(widths)
max_height = max(heights)


new_im = Image.new('RGBA', (total_width, max_height))
x_offset = 0
for im in images_point:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

new_im.save('collage_aligned_windowed.png')



new_im = Image.new('RGBA', (widths[0]*10, heights[0]*3))

y_offset = -im.size[1]
for row in range(3):
    x_offset = 0
    y_offset += im.size[1]
    for col in range(10):
        p_index = col + row * 10
        # for im in images_point:
        im = images_point[p_index]

        new_im.paste(im, (x_offset, y_offset))

        x_offset += im.size[0]

new_im.save('collage_aligned_windowed.png')

# ------------------------------------------------------------------------
# Remove temporary folder
for fn in images:
    os.remove(fn)

os.rmdir(temp_folder)




# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Montage of aligned images

# ------------------------------------------------------------------------
# Temporarily copy images from all subfolders into a single folder
temp_folder = "all-together"
os.mkdir(temp_folder)

for sub_folder in ["faces-aligned/premium", "faces-aligned/popular"]:
    listOfFiles = getAllFiles(sub_folder)
    images_A = [fn for fn in listOfFiles if os.path.splitext(fn)[1].lower() in ('.jpg', '.jpeg', '.png')]
    for fn in images_A:
        shutil.copy(fn, temp_folder)


# ------------------------------------------------------------------------
# Create collage
files = [os.path.join(temp_folder, fn) for fn in os.listdir(temp_folder)]
images = [fn for fn in files if os.path.splitext(fn)[1].lower() in ('.jpg', '.jpeg', '.png')]

images_point = [Image.open(x) for x in images]
widths, heights = zip(*(i.size for i in images_point))

total_width = sum(widths)
max_height = max(heights)


# new_im = Image.new('RGBA', (total_width, max_height))
# x_offset = 0
# for im in images_point:
#     new_im.paste(im, (x_offset,0))
#     x_offset += im.size[0]
#
# new_im.save('collage_aligned.png')



new_im = Image.new('RGBA', (widths[0]*10, heights[0]*3))

y_offset = -im.size[1]
for row in range(3):
    x_offset = 0
    y_offset += im.size[1]
    for col in range(10):
        p_index = col + row * 10
        # for im in images_point:
        im = images_point[p_index]

        new_im.paste(im, (x_offset, y_offset))

        x_offset += im.size[0]

new_im.save('collage_aligned.png')

# ------------------------------------------------------------------------
# Remove temporary folder
for fn in images:
    os.remove(fn)

os.rmdir(temp_folder)


# END
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
