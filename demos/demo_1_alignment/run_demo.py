import os
import alignfaces as af

# -----------------------------------------------------------------------------
# Template for most uses

# Path for faces directory.
my_project_path = os.path.dirname(os.path.abspath(__file__))
my_faces_path = my_project_path + os.path.sep + "faces" + os.path.sep

# Analyze all image files whose filenames have these properties ...
file_prefix = ""
file_postfix = "jpg"

# Estimate landmarks.
af.get_landmarks(my_faces_path, file_prefix, file_postfix)
#
# If landmark detection was interrupted while processing a large batch of faces
#   then run this next time instead of previous line ...
# af.get_landmarks(my_faces_path, file_prefix, file_postfix, start_fresh=False)

# Show faces with landmarks overlaid and relative file name.
#   Record file names for any inaccurate landmarks.
af.plot_faces_with_landmarks_one_by_one(my_faces_path)

# Landmark placement is accurate for all faces in this demo.
# But if there were any images with poor placement, we would create a file
# called bad-landmarks.csv with a comma-separated list of all bad file names
# like this: premium/men/corey-haim.jpg, popular/females/gigi-hadid.jpg
#
# Then we would run this:
# af.exclude_files_with_bad_landmarks(my_faces_path)

# Take stock of numbers of images excluded for either failed face detection
#   or inaccurate landmarks (as determined in previous section)
numbers = af.landmarks_report(my_faces_path, file_prefix, file_postfix)
# n0 = numbers['num_total_images']
# n1 = numbers['num_failed_detections']
# n2 = numbers['num_detected_but_removed']
# print("Total input images " + str(n0))
# print("Number of images with a failed face detection " + str(n1))
# print("Number of images with a landmark inaccuracy " + str(n2))

# Now we're ready to align the faces - via generalized Procrustes analysis.
aligned_path = af.align_procrustes(my_faces_path, file_prefix, file_postfix,
                                   color_of_result="rgb")
# af.align_procrustes(source_dir, output_dir,
#                     exclude_features=['jawline', 'left_iris', 'right_iris',
#                     'mouth_inner'], include_features=None, adjust_size='default',
#                     size_value=None, color_of_result='grayscale')

# Estimate landmarks of aligned faces.
af.get_landmarks(aligned_path, file_prefix, file_postfix)

# Just show a few aligned faces with landmarks overlaid, as a quick check
af.plot_faces_with_landmarks(aligned_path, num_faces=3)

# Set aligned faces in an aperture.
the_aperture, aperture_path = af.place_aperture(aligned_path, file_prefix,
                                                file_postfix,
                                                aperture_type="MossEgg",
                                                contrast_norm="max",
                                                color_of_result="rgb")


# -----------------------------------------------------------------------------
# A modified GPA alignment procedure that closesly matches eye-based alignment.
#   Eye-based alignment is the most typical procedure found in published papers:
#       - Scaled by distance between eyes.
#       - Translated and rotated so center of eyes share common position.
#
#   Instead of simple alignment based on center of eyes, we apply GPA to
#   all of the eye landmarks. The result is similar; perhaps more robust.

# aligned_path = my_project_path + os.path.sep + "faces-aligned-eyes" + os.path.sep
#
# # Create folders within /aligned_eyes
# af.make_files(my_faces_path, file_prefix, file_postfix, new_dir="faces-aligned-eyes")
#
# # Note that arguments provided to include_features override exclude_features.
# af.align_procrustes(my_faces_path, aligned_path, color_of_result='rgb',
#                     include_features=['left_eye', 'right_eye'])
#
# # Estimate landmarks of aligned faces.
# af.get_landmarks(aligned_path, file_prefix, file_postfix)
#
# af.plot_faces_with_landmarks(aligned_path, num_faces=3)

# -----------------------------------------------------------------------------
# For Matlab users
print("\n******************************************************************\n")
print("Are you a Matlab user?\n")
print("Is your version of Matlab >= R2016b?\n")
print("If so, you can now run >> confirm_alignment_in_matlab\n")


# END
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
