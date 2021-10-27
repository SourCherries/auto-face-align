import os
from landmark_variance import plot_landmark_dist_overlay


elip_alpha = 0.50;

my_project_path = os.path.dirname(os.path.abspath(__file__))

exc = ['jawline', 'left_iris', 'right_iris', 'mouth_inner']

databases = ["CAS-female", "CAS-male", "GUFD-female", "GUFD-male",
             "KUFD-DC", "KUFD-ID"]
file_postfixes = ["tif", "tif", "jpg", "jpg", "jpg", "jpg"]

for dbase, fp in zip(databases, file_postfixes):
    aligned_path = my_project_path + os.path.sep + dbase + "-aligned" + os.path.sep

    out_file = "landmark-dist-" + dbase + ".png"

    plot_landmark_dist_overlay(aligned_path, file_prefix='', file_postfix=fp, output_file=out_file, exclude_features=exc, elip_alpha=elip_alpha)
