import csv
import os
import io
import json

# Make into a user-accessible function.
# Only parameter should be source_dir.
# Looks for a csv (comma-separated file) named "bad-landmarks.csv".
#   The user creates this file themselves and has to name it "bad-landmarks.csv"
#   And put it into the same folder as the "landmarks.txt" file.
# Function first looks for "bad-landmarks.csv" then loads it as a list of strings.
# The rest is shown below.
source_dir = "/Users/carl/Studies/facepackage-macbook/facepackage-slim/results/KUFD-DC/"
# Put this in make_aligned_faces.py

full_bad_filename = source_dir + "bad-landmarks.csv"
exists = os.path.isfile(full_bad_filename)
if exists:
    with open(full_bad_filename) as f:
         reader = csv.reader(f)
         remove_these_files = list(reader)[0]
else:
    print(['CSV file bad-landmarks.csv does not exist in ' + source_dir])
    return

full_landmark_filename = source_dir + "landmarks.txt"
exists = os.path.isfile(full_landmark_filename)
if exists:
    with io.open(full_landmark_filename, 'r') as f:
        imported_landmarks = json.loads(f.readlines()[0].strip())
else:
    print(['JSON file landmarks.txt does not exist in ' + source_dir])
    return

# First save original with a new name just in case
full_landmark_filename = source_dir + "landmarks-original.txt"
with open(full_landmark_filename, 'w') as outfile:
    json.dump(imported_landmarks, outfile)

# Now delete unwanted entries and save as landmarks.txt
for this_file in remove_these_files:
    del imported_landmarks[this_file.strip()]
full_landmark_filename = source_dir + "landmarks.txt"
with open(full_landmark_filename, 'w') as outfile:
    json.dump(imported_landmarks, outfile)

# END
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
