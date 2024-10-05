import os
import alignfaces as af

expression = ["an", "ca", "di", "fe", "ha", "ne", "sa", "sp"]
mouth = ["o", "c"]
databases = []
for ex in expression:
    for mo in mouth:
        databases.append("NIM-" + ex + "-" + mo)
num_dirs = len(expression) * len(mouth)
file_postfixes = ["bmp"] * num_dirs
my_project_path = os.path.dirname(os.path.abspath(__file__))



# for dbase, pf in zip(databases, file_postfixes):
#     my_faces_path = my_project_path + os.path.sep + dbase + os.path.sep
#     af.get_landmarks(my_faces_path, "", pf, start_fresh=True)
#     af.plot_faces_with_landmarks_one_by_one(my_faces_path)
# exclude_files_with_bad_landmarks looks specifically for bad-landmarks.csv.
# so bad-landmarks-strict.csv will be ignored. that's good.
# bad-landmarks.csv ignore poor fits to mouth.
# i will not be using the mouth in my GPA alignment so that's good.



# for dbase, pf in zip(databases, file_postfixes):
#     my_faces_path = my_project_path + os.path.sep + dbase + os.path.sep
#     af.exclude_files_with_bad_landmarks(my_faces_path)

# AFTER DONE WITH MANUALLY RECORDING BAD LANDMARKS FOR EACH FOLDER:
#   DO COMMENTED ABOVE
#   UNCOMMENT BELOW AND RUN

# total0, total1, total2 = 0, 0, 0
# with open('table-DLIB-failures-NIM.csv', 'w') as writer:
#     writer.write("Database,n,Failed Face Detections,Inaccurate Landmarks\n")
#     for dbase, pf in zip(databases, file_postfixes):
#         my_faces_path = my_project_path + os.path.sep + dbase + os.path.sep
#         numbers = af.landmarks_report(my_faces_path, file_prefix="", file_postfix=pf)
#         n0 = numbers['num_total_images']
#         n1 = numbers['num_failed_detections']
#         n2 = numbers['num_detected_but_removed']
#         writer.write('%s,%d,%d,%d\n' % (dbase, n0, n1, n2))
#         total0 += n0
#         total1 += n1
#         total2 += n2
#     writer.write('%s,%d,%d,%d\n' % ("All", total0, total1, total2))
#
#
include_features = ["left_eye", "right_eye"]

for dbase, pf in zip(databases, file_postfixes):
    my_faces_path = my_project_path + os.path.sep + dbase + os.path.sep
    aligned_path = af.align_procrustes(my_faces_path, "", pf,
                                       color_of_result="grayscale",
                                       include_features=include_features)
    af.get_landmarks(aligned_path, "", pf, start_fresh=True)


# END
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
