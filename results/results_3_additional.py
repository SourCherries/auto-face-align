import os
import glob
import alignfaces2 as af

my_project_path = os.path.dirname(os.path.abspath(__file__))

databases = ["CAS-female", "CAS-male", "GUFD-female", "GUFD-male",
             "KUFD-DC", "KUFD-ID"]
file_postfixes = ["tif", "tif", "jpg", "jpg", "jpg", "jpg"]

# Visually inspect accuracy of landmark placement for all aligned faces.
for dbase in databases:
    aligned_path = my_project_path + os.path.sep + dbase + "-aligned" + os.path.sep
    print(aligned_path)
    af.plot_faces_with_landmarks_one_by_one(aligned_path)


# END
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
