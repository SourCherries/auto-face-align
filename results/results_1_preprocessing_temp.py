import os
import alignfaces2 as af


my_project_path = os.path.dirname(os.path.abspath(__file__))

databases = ["GUFD-female"]
file_postfixes = ["jpg"]


for dbase, pf in zip(databases, file_postfixes):
    my_faces_path = my_project_path + os.path.sep + dbase + os.path.sep
    af.get_landmarks(my_faces_path, "", pf, start_fresh=True)
    # af.plot_faces_with_landmarks_one_by_one(my_faces_path)

# print("NOTE: I created a bad-landmarks.csv file for both KUFD-DC and KUFD-ID.")
# my_faces_path = my_project_path + os.path.sep + "KUFD-DC" + os.path.sep
# af.exclude_files_with_bad_landmarks(my_faces_path)
# my_faces_path = my_project_path + os.path.sep + "KUFD-ID" + os.path.sep
# af.exclude_files_with_bad_landmarks(my_faces_path)

# total0, total1, total2 = 0, 0, 0
# with open('table-DLIB-failures.csv', 'w') as writer:
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


for dbase, pf in zip(databases, file_postfixes):
    my_faces_path = my_project_path + os.path.sep + dbase + os.path.sep
    aligned_path = af.align_procrustes(my_faces_path, "", pf, color_of_result="rgb")
    # af.get_landmarks(aligned_path, "", pf, start_fresh=True)


# END
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
