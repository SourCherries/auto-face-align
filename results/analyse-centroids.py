import alignfaces as af
import numpy as np

mother_folder = "/Users/carl/Studies/facepackage-macbook/facepackage-slim/results/"
aligned_folders = ["CAS-female-aligned/", "CAS-male-aligned/",
                   "GUFD-female-aligned/", "GUFD-male-aligned/"]

# Default values used in results_1_preprocessing.py
EF = ['jawline', 'left_iris', 'right_iris', 'mouth_inner']
MF = ['jawline', 'left_iris', 'right_iris', 'mouth_inner', 'mouth_outline']

features = ['left_eyebrow', 'left_eye', 'right_eyebrow', 'right_eye',
            'nose', 'mouth_outline', 'mouth_inner', 'jawline',
            'left_iris', 'right_iris']
IE = list(set(features) - set(['left_eye', 'left_eye']))


all_delta = []
all_delta_eyes_only = []

for this_folder in aligned_folders:
    source_dir = mother_folder + this_folder

    # Centroids for original alignment using default landmarks.
    landmark_features, files = af.get_landmark_features(source_dir, output_dir="",
                                                        exclude_features=EF)
    centroids_original = np.array(landmark_features["MeanXY"])
    (nf_original, _) = centroids_original.shape

    # Centroids for default minus mouth.
    landmark_features, files = af.get_landmark_features(source_dir, output_dir="",
                                                        exclude_features=MF)
    centroids_top = np.array(landmark_features["MeanXY"])
    (nf_top, _) = centroids_top.shape

    # Centroids for eyes only.
    landmark_features, files = af.get_landmark_features(source_dir, output_dir="",
                                                        exclude_features=IE)
    centroids_eyes = np.array(landmark_features["MeanXY"])

    # Simple check.
    assert (nf_original-nf_top)==0

    delta = (centroids_original[:,1] - centroids_top[:,1]).mean()
    print(this_folder + "\t(original - exclude_mouth) = " + str(delta) + "\n")

    delta_eyes_only = (centroids_original[:,1] - centroids_eyes[:,1]).mean()
    print(this_folder + "\t(original - eyes_only) = " + str(delta_eyes_only) + "\n")

    all_delta.append(delta)
    all_delta_eyes_only.append(delta_eyes_only)


print("Mean\t(original - exclude_mouth) = " + str(np.array(all_delta).mean()) + "\n")

print("Mean\t(original - eyes_only) = " + str(np.array(all_delta_eyes_only).mean()) + "\n")
# END
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
