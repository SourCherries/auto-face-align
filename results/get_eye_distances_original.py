import os
import alignfaces2 as af
import numpy as np


my_project_path = os.path.dirname(os.path.abspath(__file__))
databases = ["CAS-female", "CAS-male", "GUFD-female", "GUFD-male",
             "KUFD-DC", "KUFD-ID"]

with open('table-eye-distances.csv', 'w') as writer:
    total_n = 0;
    total_failed = 0;
    print("Database", "Mean Eye Distance (pixels)", sep=",")
    writer.write("Database,Mean Eye Distance (pixels)\n")
    for dbase in databases:
        my_faces_path = my_project_path + os.path.sep + dbase + os.path.sep

        landmark_features, files = af.get_landmark_features(my_faces_path)
        this_mean_dist = round(np.array(landmark_features['eye_distances']).mean())

        print(dbase, this_mean_dist, sep=",")
        writer.write('%s,%d\n' % (dbase, this_mean_dist))



# END
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
