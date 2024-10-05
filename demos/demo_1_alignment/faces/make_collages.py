import os
import shutil
import random

from collage_maker import make_collage

# ------------------------------------------------------------------------
# Helper function
def getAllFiles(dirName):
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    return listOfFiles

# ------------------------------------------------------------------------
# Temporarily copy images from all subfolders into a single folder
temp_folder = "all-together"
os.mkdir(temp_folder)

for sub_folder in ["premium", "popular"]:
    listOfFiles = getAllFiles(sub_folder)
    images_A = [fn for fn in listOfFiles if os.path.splitext(fn)[1].lower() in ('.jpg', '.jpeg', '.png')]
    for fn in images_A:
        shutil.copy(fn, temp_folder)

# listOfFiles = getAllFiles("popular")
# images_B = [fn for fn in listOfFiles if os.path.splitext(fn)[1].lower() in ('.jpg', '.jpeg', '.png')]
# for fn in images_B:
#     shutil.copy(fn, temp_folder)

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

# END
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
