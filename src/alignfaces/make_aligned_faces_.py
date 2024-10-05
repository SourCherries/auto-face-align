from .face_landmarks import (facial_landmarks, pull_jawline_to_inside_of_hairline,
                            unpack_dict_to_vector_xy, pack_vector_xy_as_dict)

from math import floor, ceil

# import matplotlib.pyplot as plt
import numpy as np
import math
import os.path

# image io and grayscale conversion
from PIL import Image as PilImage

# image translation and rotation
from skimage.transform import SimilarityTransform, warp

# image scaling
import dlib

# Procrustes analysis
# import procrustesTool as pt
# from .procrustes_tools import procrustes_tools as pt
from .procrustes_tools import (get_translation,
                              generalized_procrustes_analysis,
                              translate, get_rotation_scale,
                              procrustes_analysis)

# from .apertureTools import (fit_ellipse_semi_minor, make_ellipse_map,
#                             make_circle_map, make_map_below_y,
#                             make_moss_egg, make_four_channel_image)

from .aperture_tools import (fit_ellipse_semi_minor, make_ellipse_map,
                            make_moss_egg, make_four_channel_image)

from .warp_tools import pawarp

from .contrast_tools import contrast_stretch

from .plot_tools import (plot_faces_with_landmarks,
                         plot_faces_with_landmarks_one_by_one,
                         fit_ellipse_to_2D_cloud,
                         plot_ellipse)

# import pdb  # TEMPORARY FOR DEBUGGING
# Final adjustments and saving image
from skimage import exposure
from skimage.io import imsave

# files
import pickle
import os
import json
import io
import csv

# testing
import sys

# additional functions
from .make_files import get_source_files, clone_directory_tree

# adjust_size = 'default'
# size_value = None
# start_fresh = True
# iris_by_coherence = False

# -----------------------------------------------------------------------------
# Support function
def get_rotation_matrix_2d(center, angle):
    (x0, y0) = center[0], center[1]
    M = np.array([[np.cos(angle), -np.sin(angle),
                   x0*(1 - np.cos(angle)) + y0*np.sin(angle)],
                  [np.sin(angle),  np.cos(angle),
                   y0*(1 - np.cos(angle)) - x0*np.sin(angle)],
                  [0, 0, 1]])
    return M

# -----------------------------------------------------------------------------
# adjust_size, size_value -> only in 1st section, which is not necessary here.
# color_of_result -> 303
#
# actually required:
#   MotherDir, start_fresh, iris_by_coherence


def get_landmarks(MotherDir, file_prefix, file_postfix, include_jaw=False,
                  start_fresh=True):

    # files = make_files(MotherDir, file_prefix, file_postfix)
    # source_files = files[0]
    # output_files = files[1]
    source_files = get_source_files(MotherDir, FilePrefix=file_prefix,
                                    FilePostfix=file_postfix)

    # adjust_size='default', size_value=None, color_of_result='grayscale'

    # MotherDir = "/Users/carl/Studies/Collaborations/Eugenie/RawFacesCleaned/"
    # MotherDir = "/Users/carl/Studies/your_project_here/faces/"
    # viewpoint = 'all'  # 'front', 'left', 'right'

    # Copy directory structure for output
    MotherBits = MotherDir.split(os.path.sep)
    go_back = -len(MotherBits[-2]) - 1
    GrannyDir = MotherDir[0:go_back]
    file_results_temporary = MotherDir + "results_temporary.dat"
    file_results = MotherDir + "results.dat"

    if start_fresh:
        try:
            with open(file_results_temporary, "rb") as f:
                print(file_results_temporary + "\t exists.\nRemoving ...")
            os.remove(file_results_temporary)
        except IOError as error:
            print(error)

    # To do: User interface for getting this, or simple arguments into script
    keys_to_remove_for_alignment = ['JAWLINE_POINTS']

    # try:
    #     f = open(file_results_aligned, "rb")
    #     data2 = []
    #     for _ in range(pickle.load(f)):
    #         data2.append(pickle.load(f))
    #     LabelToCoordinateIndex = data2[0]
    #     output_files = data2[1]
    #     FinalLandmarks = data2[2]
    #     f.close()
    # except:
    ###########################################################################
    # input and output files
    # with open(GrannyDir + 'input_files.data', 'rb') as filehandle:
    #     source_files = pickle.load(filehandle)
    # with open(GrannyDir + 'output_files.data', 'rb') as filehandle:
    #     output_files = pickle.load(filehandle)

    ###########################################################################
    # Main

    # Obtain facial landmarks and center-point for all faces
    try:
        with open(file_results_temporary, "rb") as f:
            data_in = []
            for _ in range(pickle.load(f)):
                data_in.append(pickle.load(f))
            AllSubjectsLandmarksDict = data_in[0]
            AllSubjectsLandmarksDictAllFeature = data_in[1]
            MeanXY = data_in[2]
            LandmarkFailureFile = data_in[3]
            LandmarkFailureIndex = data_in[4]
            StartingIndex = len(MeanXY) + len(LandmarkFailureFile)
            IrisPoints = data_in[5]
    except IOError as error:
        print(error)
        AllSubjectsLandmarksDict = []
        AllSubjectsLandmarksDictAllFeature = []
        MeanXY = []
        LandmarkFailureFile = []
        LandmarkFailureIndex = []
        StartingIndex = 0
        IrisPoints = []

    # for infile, outfile, facei in zip(source_files[StartingIndex:], output_files[StartingIndex:], range(StartingIndex, len(source_files))):
    for infile, facei in zip(source_files[StartingIndex:], range(StartingIndex, len(source_files))):
        # image = cv2.imread(infile)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.array(PilImage.open(infile).convert("L"))
        Landmarks = facial_landmarks(gray)

        if Landmarks is None:
            print("Landmarks is None: " + str(facei))
            LandmarkFailureFile.append(infile)
            LandmarkFailureIndex.append(facei)
            # AllSubjectsLandmarksDictAllFeature.append(np.array(CoordinateVector))
            #
            # AllSubjectsLandmarksDict.append(np.array(CoordinateVector))
            #
            # MeanXY.append(np.array([None, None]))
        else:
            le = Landmarks['LEFT_EYE_POINTS']
            re = Landmarks['RIGHT_EYE_POINTS']
            mle = le.mean(axis=0)
            mre = re.mean(axis=0)
            temp_dict = {0: (int(mre[1]), int(mre[0])),
                         1: (int(mle[1]), int(mle[0]))}
            IrisPoints.append(temp_dict)

            Landmarks = pull_jawline_to_inside_of_hairline(Landmarks, gray)
            CoordinateVector, LabelToCoordinateIndexAll = unpack_dict_to_vector_xy(Landmarks)
            AllSubjectsLandmarksDictAllFeature.append(np.array(CoordinateVector))

            for this_key in keys_to_remove_for_alignment:
                Landmarks.pop(this_key, None)

            CoordinateVector, LabelToCoordinateIndex = unpack_dict_to_vector_xy(Landmarks)
            AllSubjectsLandmarksDict.append(np.array(CoordinateVector))

            mean_x, mean_y = get_translation(CoordinateVector)
            MeanXY.append(np.array([mean_x, mean_y]))

        f = open(file_results_temporary, "wb")
        data = [AllSubjectsLandmarksDict, AllSubjectsLandmarksDictAllFeature, MeanXY, LandmarkFailureFile, LandmarkFailureIndex, IrisPoints]
        pickle.dump(len(data), f)
        for value in data:
            pickle.dump(value, f)
        f.close()

        print(f'{facei:5d}', end="\r")

    # Remove files with failed landmark-detection.
    # for bi, bf in zip(LandmarkFailureIndex, LandmarkFailureFile):
    #     print("Removing " + source_files[bi])
    #     print("     which is " + bf)
    #     del source_files[bi]
    #     del output_files[bi]
    SF = np.asarray(source_files)
    # OF = np.asarray(output_files)
    # keep_locs = (set([l for l in range(len(output_files))]) -
    #              set(LandmarkFailureIndex))
    keep_locs = (set([l for l in range(len(source_files))]) -
                 set(LandmarkFailureIndex))
    source_files = list(SF[list(keep_locs)])
    # output_files = list(OF[list(keep_locs)])

    # Rewrite those files to disk (source directory)
    # with open(MotherDir + 'input_files.data', 'wb') as filehandle:
    #     pickle.dump(source_files, filehandle)
    # with open(MotherDir + 'output_files.data', 'wb') as filehandle:
    #     pickle.dump(output_files, filehandle)

    # Create new file that lists any failures
    # with open(MotherDir + 'excluded_files.data', 'wb') as filehandle:
    #     pickle.dump(LandmarkFailureFile, filehandle)

    # -----------------------------------------------------------------------------
    # Produce a single JSON file that contains all landmark data
    lmd = len(MotherDir)
    # Combine into single dictionary
    features = ['LEFT_EYEBROW', 'LEFT_EYE',
                'RIGHT_EYEBROW', 'RIGHT_EYE',
                'NOSE', 'MOUTH_OUTLINE',
                'MOUTH_INNER', 'JAWLINE']
    landmarks = dict()
    number_photos = len(source_files)
    for outfile, i in zip(source_files, range(0, number_photos)):
        # LandmarksThisGuy = FinalLandmarks[i]
        LandmarksThisGuy = AllSubjectsLandmarksDictAllFeature[i]
        ThisSubjectLandmarksDict = pack_vector_xy_as_dict(LandmarksThisGuy.tolist(),
                                                      LabelToCoordinateIndexAll)
        fullpath = source_files[i]
        # partpath = fullpath[fullpath.index('aligned') + len('aligned'):]
        partpath = fullpath[lmd:]
        this_photo = dict()
        for fi in range(len(features)):
            this_array = ThisSubjectLandmarksDict[features[fi] + '_POINTS']
            # this_array = AllSubjectsLandmarksDictAllFeature[features[fi] + '_POINTS']
            this_photo[features[fi].lower()] = \
                list(this_array.flatten().astype(float))
                # list(this_array.flatten().astype(np.float))
        # TO DO
        #   append iris points [x, y] to this_photo
        # take from IrisPoints
        #
        # IrisPoints is list [p1, p2, ... pn] for n people, where
        #   p_i is dictionary
        #       p_i{0} = (y, x)_for right eye
        #       p_i{1} = (y, x) for left eye
        this_photo['left_iris'] = [float(IrisPoints[i][1][1]),
                                   float(IrisPoints[i][1][0])]
        this_photo['right_iris'] = [float(IrisPoints[i][0][1]),
                                    float(IrisPoints[i][0][0])]
        landmarks[partpath] = this_photo

    # export to JSON
    json_file = MotherDir + 'landmarks.txt'
    with open(json_file, 'w') as outfile:
        json.dump(landmarks, outfile)

    # End of chunk for Getting Landmarks
    #
    # Output:
    # source_files, output_files, AllSubjectsLandmarksDict,
    # AllSubjectsLandmarksDictAllFeature, MeanXY, LandmarkFailureFile,
    # LandmarkFailureIndex, IrisPoints
    # files = []
    # files.append(source_files)
    # files.append(output_files)
    if include_jaw:
        listOfLandmarks = AllSubjectsLandmarksDictAllFeature
    else:
        listOfLandmarks = AllSubjectsLandmarksDict
    print("\n\n------------------------------------------------------------\n")
    print("\nThere should now be a JSON-formatted" +
          "file called landmarks.txt in:\n")
    print("\t" + MotherDir + "\n")
    print("Please examine the contents.\n\n")
    print("\n--------------------------------------------------------------\n")
    return source_files, listOfLandmarks, MeanXY, IrisPoints
    # -------------------------------------------------------------------------


def get_landmark_features(source_dir, output_dir="", exclude_features=['jawline',
                          'left_iris', 'right_iris', 'mouth_inner']):
    # Returns a DICT with lots of useful stuff, like eye_distances.
    # AllSubjectsLandmarksDict is landmarks for all faces.
    # exclude_features refers to this; does not affect calculation of other stuff.

    # Determine if landmarks.txt exists in source_dir.
    # Load or return error if does not exist.
    full_landmark_filename = source_dir + 'landmarks.txt'
    exists = os.path.isfile(full_landmark_filename)
    if exists:
        with io.open(full_landmark_filename, 'r') as f:
            imported_landmarks = json.loads(f.readlines()[0].strip())
    else:
        print(['JSON file landmarks.txt does not exist in ' + source_dir])
        return

    # Initialize results
    eye_distances = []
    eye_radians = []
    eye_midpoints = []
    mouth2eye_distances = []
    mouth2eye_radians = []
    nose_tips = []

    # Combine source_dir and output_dir with keys of imported_landmarks to get
    #   source_files and output_files.
    #
    # Also obtaining corresponding LOL as appropriate
    #   input to alignment functions.
    source_files = []
    output_files = []
    LOL = []
    MeanXY = []
    IrisPoints = []
    for rest_of_file_name in imported_landmarks.keys():
        source_files.append(source_dir + rest_of_file_name)
        output_files.append(output_dir + rest_of_file_name)

        # Convert imported_landmarks -> LOL (like listOfLandmarks)
        dict_of_coordinates = imported_landmarks[rest_of_file_name]
        this_person = np.array([], dtype=np.int64).reshape(0,)
        this_iris = dict()

        # Get specific features.
        dickeys = dict_of_coordinates.keys()
        if ('left_iris' in dickeys) and ('right_iris' in dickeys):
            left_iris = np.array(dict_of_coordinates['left_iris'])
            right_iris = np.array(dict_of_coordinates['right_iris'])
        else:
            ex = np.array(dict_of_coordinates['left_eye'][0::2]).mean()
            ey = np.array(dict_of_coordinates['left_eye'][1::2]).mean()
            left_iris = np.array([ex, ey])
            ex = np.array(dict_of_coordinates['right_eye'][0::2]).mean()
            ey = np.array(dict_of_coordinates['right_eye'][1::2]).mean()
            right_iris = np.array([ex, ey])

        mouth_inner = dict_of_coordinates['mouth_inner']
        mouth_outline = dict_of_coordinates['mouth_outline']
        nx = dict_of_coordinates['nose'][0::2]
        ny = dict_of_coordinates['nose'][1::2]
        NOSE_TIP = np.array([nx[6], ny[6]])
        nose_tips.append(NOSE_TIP)

        eye_vector = left_iris - right_iris
        EYE_DIST = np.sqrt(((eye_vector)**2).sum())
        EYE_RAD = np.arctan2(eye_vector[1], eye_vector[0])
        trans_eye = np.array([(EYE_DIST / 2) * np.cos(EYE_RAD),
                             (EYE_DIST / 2) * np.sin(EYE_RAD)])
        EYE_MID = right_iris + trans_eye
        eye_distances.append(EYE_DIST)
        eye_radians.append(EYE_RAD)
        eye_midpoints.append(EYE_MID)

        mix = np.array(mouth_inner[0::2]).mean()
        miy = np.array(mouth_inner[1::2]).mean()
        mox = np.array(mouth_outline[0::2]).mean()
        moy = np.array(mouth_outline[1::2]).mean()
        mcx, mcy = (mox + mix) / 2, (moy + miy) / 2
        mouth = np.array([mcx, mcy])

        mouth_vector = mouth - EYE_MID
        MOUTH_DIST = np.sqrt(((mouth_vector)**2).sum())
        MOUTH_RAD = np.arctan2(mouth_vector[1], mouth_vector[0])
        mouth2eye_distances.append(MOUTH_DIST)
        mouth2eye_radians.append(MOUTH_RAD)

        # Concatenation of landmarks (with some exclusions)
        for this_feature in dict_of_coordinates.keys():
            if not (this_feature in exclude_features):
                these_coords = dict_of_coordinates[this_feature]
                this_person = np.hstack((this_person, np.array(these_coords)))
            if (this_feature == 'left_iris'):
                these_coords = dict_of_coordinates[this_feature]
                this_iris[1] = np.array([these_coords[1], these_coords[0]])
            if (this_feature == 'right_iris'):
                these_coords = dict_of_coordinates[this_feature]
                this_iris[0] = np.array([these_coords[1], these_coords[0]])
        LOL.append(this_person)
        IrisPoints.append(this_iris)
        xj = this_person[0::2]
        yj = this_person[1::2]
        MeanXY.append(np.array([int(xj.mean()), int(yj.mean())]))

    AllSubjectsLandmarksDict = LOL

    # Vectors of indices for each facial feature
    for remove_this_feature in exclude_features:
        del dict_of_coordinates[remove_this_feature]
    CoordinateVector, LabelToCoordinateIndex = unpack_dict_to_vector_xy(dict_of_coordinates)

    # Package final results.
    landmark_features = dict()
    landmark_features['eye_distances'] = eye_distances
    landmark_features['eye_radians'] = eye_radians
    landmark_features['eye_midpoints'] = eye_midpoints
    landmark_features['mouth2eye_distances'] = mouth2eye_distances
    landmark_features['mouth2eye_radians'] = mouth2eye_radians
    landmark_features['nose_tips'] = nose_tips

    landmark_features['AllSubjectsLandmarksDict'] = AllSubjectsLandmarksDict
    landmark_features['IrisPoints'] = IrisPoints
    landmark_features['MeanXY'] = MeanXY
    landmark_features['LabelToCoordinateIndex'] = LabelToCoordinateIndex

    files = []
    files.append(source_files)
    files.append(output_files)
    return landmark_features, files


def landmarks_report(source_dir, file_prefix="", file_postfix="jpg"):
    numbers = {}
    infiles = get_source_files(source_dir, file_prefix, file_postfix)
    rel_paths = [fp[len(source_dir):] for fp in infiles]
    num_total_images = len(rel_paths)

    full_path_inaccurate = source_dir + "bad-landmarks.csv"
    if os.path.isfile(full_path_inaccurate):
        with open(full_path_inaccurate) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                num_detected_but_removed = len(row);
    else:
        num_detected_but_removed = 0;

    landmark_features, files = get_landmark_features(source_dir, source_dir)
    num_in_landmarks_file = len(files[0])
    num_failed_detections = (num_total_images - num_in_landmarks_file -
                             num_detected_but_removed)
    numbers['num_total_images'] = num_total_images
    numbers['num_detected_but_removed'] = num_detected_but_removed
    numbers['num_failed_detections'] = num_failed_detections

    print("Total input images " + str(num_total_images))
    print("Number of images with a failed face detection " + str(num_failed_detections))
    print("Number of images with a landmark inaccuracy " + str(num_detected_but_removed))

    return numbers


def exclude_files_with_bad_landmarks(source_dir):
    full_bad_filename = source_dir + "bad-landmarks.csv"
    exists = os.path.isfile(full_bad_filename)
    if exists:
        with open(full_bad_filename) as f:
             reader = csv.reader(f)
             remove_these_files = list(reader)[0]
    else:
        print(['CSV file bad-landmarks.csv does not exist in ' + source_dir])
        return None

    full_landmark_filename = source_dir + "landmarks.txt"
    exists = os.path.isfile(full_landmark_filename)
    if exists:
        with io.open(full_landmark_filename, 'r') as f:
            imported_landmarks = json.loads(f.readlines()[0].strip())
    else:
        print(['JSON file landmarks.txt does not exist in ' + source_dir])
        return None

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
    return None


def tilt_radians_to_ensure_eyes_level(the_shape, LabelToCoordinateIndex):
    x_locs = LabelToCoordinateIndex["left_eye"][::2]
    y_locs = LabelToCoordinateIndex["left_eye"][1::2]
    left_center = np.array([the_shape[x_locs].mean(), the_shape[y_locs].mean()])
    x_locs = LabelToCoordinateIndex["right_eye"][::2]
    y_locs = LabelToCoordinateIndex["right_eye"][1::2]
    right_center = np.array([the_shape[x_locs].mean(), the_shape[y_locs].mean()])
    eye_vector = left_center - right_center
    clock_wise_norm = (-eye_vector[1], eye_vector[0])
    eye_tilt_radians = np.arctan2(clock_wise_norm[1], clock_wise_norm[0])
    rotate_to_level_radians = np.pi / 2 - eye_tilt_radians
    return rotate_to_level_radians


def align_procrustes(source_dir, file_prefix='', file_postfix='jpg',
                     exclude_features=['jawline', 'left_iris', 'right_iris',
                     'mouth_inner'], include_features=None,
                     adjust_size='default', size_value=None,
                     color_of_result='grayscale'):
    # NOTE: include_features overrides exclude_features

    # files, output_dir = make_files(source_dir, file_prefix, file_postfix,
    #                                new_dir="aligned")
    # source_files = files[0]
    # output_files = files[1]

    output_dir = clone_directory_tree(source_dir, new_dir="aligned",
                                      FilePrefix=file_prefix,
                                      FilePostfix=file_postfix)

    if include_features != None:
        features = ['left_eyebrow', 'left_eye', 'right_eyebrow', 'right_eye',
                    'nose', 'mouth_outline', 'mouth_inner', 'jawline',
                    'left_iris', 'right_iris']
        exclude_features = list(set(features) - set(include_features))

    landmark_features, files = get_landmark_features(source_dir, output_dir,
                                                     exclude_features)
    source_files = files[0]
    output_files = files[1]
    # if len((set(source_files) - set(source_files_))) !=0:
    #     print("source_files returned by make_files() and get_landmark_features() are different.")
    #     print(source_files[0] + "\n" + source_files_[0])
    #     return
    # source_files = files[0]
    # output_files = files[1]
    AllSubjectsLandmarksDict = landmark_features['AllSubjectsLandmarksDict']
    IrisPoints = landmark_features['IrisPoints']
    MeanXY = landmark_features['MeanXY']

    LabelToCoordinateIndex = landmark_features['LabelToCoordinateIndex']

    # # Determine if landmarks.txt exists in source_dir.
    # # Load or return error if does not exist.
    # full_landmark_filename = source_dir + 'landmarks.txt'
    # exists = os.path.isfile(full_landmark_filename)
    # if exists:
    #     with io.open(full_landmark_filename, 'r') as f:
    #         imported_landmarks = json.loads(f.readlines()[0].strip())
    # else:
    #     print(['JSON file landmarks.txt does not exist in ' + source_dir])
    #     return
    #
    # # Combine source_dir and output_dir with keys of imported_landmarks to get
    # #   source_files and output_files.
    # #
    # # Also obtaining corresponding LOL as appropriate input
    # # to alignment functions.
    # # exclude_features = ['jawline', 'left_iris', 'right_iris']
    # source_files = []
    # output_files = []
    # LOL = []
    # MeanXY = []
    # IrisPoints = []
    # for rest_of_file_name in imported_landmarks.keys():
    #     source_files.append(source_dir + rest_of_file_name)
    #     output_files.append(output_dir + rest_of_file_name)
    #
    #     # Convert imported_landmarks -> LOL (like listOfLandmarks)
    #     dict_of_coordinates = imported_landmarks[rest_of_file_name]
    #     this_person = np.array([], dtype=np.int64).reshape(0,)
    #     this_iris = dict()
    #     for this_feature in dict_of_coordinates.keys():
    #         if not (this_feature in exclude_features):
    #             these_coords = dict_of_coordinates[this_feature]
    #             this_person = np.hstack((this_person, np.array(these_coords)))
    #         if (this_feature == 'left_iris'):
    #             these_coords = dict_of_coordinates[this_feature]
    #             this_iris[1] = (these_coords[1], these_coords[0])
    #         if (this_feature == 'right_iris'):
    #             these_coords = dict_of_coordinates[this_feature]
    #             this_iris[0] = (these_coords[1], these_coords[0])
    #     LOL.append(this_person)
    #     IrisPoints.append(this_iris)
    #     xj = this_person[0::2]
    #     yj = this_person[1::2]
    #     MeanXY.append(np.array([int(xj.mean()), int(yj.mean())]))
    #
    # AllSubjectsLandmarksDict = LOL

    # IrisPoints is list [p1, p2, ... pn] for n people, where
    #   p_i is dictionary
    #       p_i{0} = (y, x)_for right eye
    #       p_i{1} = (y, x) for left eye

    # -------------------------------------------------------------------------
    # EYE_DISTANCE, IMAGE_WIDTH, IMAGE_HEIGHT not used until 269-323
    #   EYE_DISTANCE -> Procrustes scaling of template 269
    #   IMAGE_WIDTH, IMAGE_HEIGHT -> padding/cropping 322-23
    #
    # rest of variables within are local to this section.
    assert adjust_size in ['set_eye_distance', 'set_image_width',
                           'set_image_height', 'default']
    if adjust_size == 'default':
        assert size_value is None

    #  values for setting fixed proportions
    _EYE_DISTANCE = 94
    _IMAGE_WIDTH = 300 - 20
    _IMAGE_HEIGHT = 338 + 80 - 30

    if adjust_size == 'default':
        EYE_DISTANCE = _EYE_DISTANCE
        IMAGE_WIDTH = _IMAGE_WIDTH
        IMAGE_HEIGHT = _IMAGE_HEIGHT
    elif adjust_size == 'set_eye_distance':
        EYE_DISTANCE = size_value
        IMAGE_WIDTH = round((_IMAGE_WIDTH / _EYE_DISTANCE) * EYE_DISTANCE)
        IMAGE_HEIGHT = round((_IMAGE_HEIGHT / _EYE_DISTANCE) * EYE_DISTANCE)
    elif adjust_size == 'set_image_width':
        IMAGE_WIDTH = size_value
        EYE_DISTANCE = round((_EYE_DISTANCE / _IMAGE_WIDTH) * IMAGE_WIDTH)
        IMAGE_HEIGHT = round((_IMAGE_HEIGHT / _IMAGE_WIDTH) * IMAGE_WIDTH)
    elif adjust_size == 'set_image_height':
        IMAGE_HEIGHT = size_value
        EYE_DISTANCE = round((_EYE_DISTANCE / _IMAGE_HEIGHT) * IMAGE_HEIGHT)
        IMAGE_WIDTH = round((_IMAGE_WIDTH / _IMAGE_HEIGHT) * IMAGE_HEIGHT)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Get eyedist and jawdist.
    # Need IrisPoints and AllSubjectsLandmarksDictAllFeature.

    # NOTES ON ORIGINAL FORMAT OF IrisPoints
    #
    # IrisPoints is list [p1, p2, ... pn] for n people, where
    #   p_i is dictionary
    #       p_i{0} = (y, x)_for right eye
    #       p_i{1} = (y, x) for left eye

    number_so_far = len(IrisPoints)
    # jawdist = np.empty((number_so_far, 1))
    eyedist = np.empty((number_so_far, 1))
    for this_face in range(0, number_so_far):
        # lx = AllSubjectsLandmarksDictAllFeature[this_face][0::2]
        # ly = AllSubjectsLandmarksDictAllFeature[this_face][1::2]
        ey = [IrisPoints[this_face][0][0], IrisPoints[this_face][1][0]]
        ex = [IrisPoints[this_face][0][1], IrisPoints[this_face][1][1]]
        # JX = lx[0:17:16]
        # JY = ly[0:17:16]
        # JD = abs(np.diff(JX + JY * 1j))
        ED = abs(np.diff(np.array(ex) + np.array(ey) * 1j))
        # jawdist[this_face, 0] = JD[0]
        eyedist[this_face, 0] = ED[0]
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Generalised Procrustes analysis
    #
    # Need:
    #   AllSubjectsLandmarksDict
    #   eyedist, EYE_DISTANCE
    #   source_files, output_files
    #   color_of_result
    #   IMAGE_WIDTH, IMAGE_HEIGHT
    #
    # To do:
    #   Rotate mean_shape_proper_scale by
    #       (a) line between mouth-center and midpoint of eyes.
    #       (b) line between eyes
    #       (c) leave as is
    #
    #   Scale
    #       (a) line between mouth-center and midpoint of eyes.
    #       (b) line between eyes [CURRENT]
    #
    #   Actually, the use of specialized GP function is not necessary hereself.
    #   I simply need the mean shape, which is simple to calculate and whose
    #   values for each landmark are independent.
    #
    #   I acutally code the rest myself without specialized GP.
    #   Check to see if that's actually correct.
    #
    #   True but:
    #       new_shapes is useful for comparing with landmarks of aligned images.
    # mean_shape, new_shapes = pt.generalized_procrustes_analysis(AllSubjectsLandmarksDict)
    mean_shape, new_shapes, shit = generalized_procrustes_analysis(AllSubjectsLandmarksDict)
    # pdb.set_trace()  # TEMPORARY FOR DEBUGGING
    # # mean_shape_with_jawline, variable_not_used = pt.generalized_procrustes_analysis(AllSubjectsLandmarksDictAllFeature)
    # mean_shape_with_jawline, shit, shat = pt.generalized_procrustes_analysis(AllSubjectsLandmarksDictAllFeature)
    #
    # # Set mean shape to have a head width desired for all of the faces
    # LandmarksOfCenteredMeanShape = pack_vector_xy_as_dict(mean_shape_with_jawline.tolist(), LabelToCoordinateIndexAll)
    # HeadWidthOfMeanShape = LandmarksOfCenteredMeanShape['JAWLINE_POINTS'][:,0].max() - LandmarksOfCenteredMeanShape['JAWLINE_POINTS'][:,0].min()


    # LandmarksOfCenteredMeanShape = pack_vector_xy_as_dict(mean_shape.tolist(), LabelToCoordinateIndex)
    # LE = LandmarksOfCenteredMeanShape['LEFT_EYE_POINTS'].mean(axis=0)
    # RE = LandmarksOfCenteredMeanShape['RIGHT_EYE_POINTS'].mean(axis=0)
    # EyeDistOfMeanShape = np.sqrt(((RE-LE)**2).sum())
    # print([EyeDistOfMeanShape, MeanEyeDist])


    # EYE_DISTANCE = HeadWidthConstant * EyeDistToHeadWidthRatio

    # Incorrect denominator for normalization:
    # MeanEyeDist = eyedist.mean()
    # mean_shape_proper_scale = mean_shape * (EYE_DISTANCE / MeanEyeDist)
    # or course, it should be the actual distance within mean_shape!!!

    have_both_eyes = ("right_eye" in list(LabelToCoordinateIndex) and
                      "left_eye" in list(LabelToCoordinateIndex))

    if have_both_eyes:
        left_eye = mean_shape[LabelToCoordinateIndex["left_eye"]]
        right_eye = mean_shape[LabelToCoordinateIndex["right_eye"]]
        LX = left_eye[::2].mean()
        LY = left_eye[1::2].mean()
        RX = right_eye[::2].mean()
        RY = right_eye[1::2].mean()
        mean_shape_eye_dist = np.sqrt((LX-RX)**2 + (LY-RY)**2)
        mean_shape_proper_scale = mean_shape * (EYE_DISTANCE / mean_shape_eye_dist)
        LabelToCoordinateIndex = landmark_features['LabelToCoordinateIndex']
        angle_constant_rad = tilt_radians_to_ensure_eyes_level(mean_shape_proper_scale, LabelToCoordinateIndex)
        angle_constant = angle_constant_rad * 180 / math.pi
    else:
        mean_shape_proper_scale = mean_shape
        angle_constant = 0

    # Now obtain precise scale and angle change for AllSubjectsLandmarksDict[i] --> mean_shape_proper_scale
    # FinalLandmarks = []
    # FinalLandmarksLimited = []
    # LastLandmarkFailureFile = []
    # LastLandmarkFailureIndex = []
    current_working_dir = os.getcwd()
    for infile, outfile, CenterOnThisFace in zip(source_files, output_files, range(0,len(source_files))):
    # for infile, outfile, CenterOnThisFace in zip(source_files[0:50], output_files[0:50], range(0, 50)):

        ################################
        # from procrustes_analysis.py
        temp_sh = np.copy(AllSubjectsLandmarksDict[CenterOnThisFace])
        translate(temp_sh)

        # get scale and rotation
        scale, theta = get_rotation_scale(mean_shape_proper_scale, temp_sh)
        scale = 1 / scale
        angle_degree = (theta * 180 / math.pi) + angle_constant
        ################################

        # 0. load image to be aligned
        # image = cv2.imread(infile)
        image = np.array(PilImage.open(infile))

        if color_of_result == 'grayscale':
            # input_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            input_image = np.array(PilImage.open(infile).convert("L"))
        else:
            input_image = image

        # 1. translate center of landmarks to center of image
        rows, cols = input_image.shape[0], input_image.shape[1]
        Cy, Cx = rows/2, cols/2
        DeltaX = Cx - MeanXY[CenterOnThisFace][0]
        DeltaY = Cy - MeanXY[CenterOnThisFace][1]
        # M = np.float32([[1, 0, DeltaX], [0, 1, DeltaY]])
        # translated_image = cv2.warpAffine(input_image, M, (cols, rows))
        tform = SimilarityTransform(translation=(-DeltaX, -DeltaY))
        translated_image = warp(input_image, tform, preserve_range=True)

        # 2. rotate to match orientation of mean shape
        # cv_rotation_matrix = cv2.get_rotation_matrix_2d((Cx, Cy), -angle_degree, 1)
        # rotated_image = cv2.warpAffine(translated_image,
        #                                cv_rotation_matrix, (cols, rows))
        angle_radians = angle_degree * np.pi / 180
        M = get_rotation_matrix_2d((Cx, Cy), -angle_radians)
        tform = SimilarityTransform(matrix=M)
        rotated_image = warp(translated_image, tform, preserve_range=True).astype(np.uint8)

        # 3. scale to match size of mean shape & recalculate image center
        # scaled_image = cv2.resize(rotated_image, (0, 0), fx=scale, fy=scale)
        oldrows, oldcols = rotated_image.shape[0], rotated_image.shape[1]
        scaled_image = dlib.resize_image(rotated_image, int(oldrows * scale),
                                         int(oldcols * scale))

        DeltaY = EYE_DISTANCE * 5 / 12  # Can add additional options here
        tform = SimilarityTransform(translation=(0, -DeltaY))
        cropped_image = warp(scaled_image, tform, preserve_range=True).astype(np.uint8)

        # Additional shift if mouth landmarks excluded from alignment.
        ema = "mouth_inner" in exclude_features
        emb = "mouth_outline" in exclude_features
        if (ema & emb):
            DeltaY = EYE_DISTANCE * 29 / 94
            tform = SimilarityTransform(translation=(0, DeltaY))
            cropped_image = warp(cropped_image, tform, preserve_range=True).astype(np.uint8)

        rows, cols = cropped_image.shape[0], cropped_image.shape[1]
        Cy, Cx = rows/2, cols/2

        # 4. crop to be 256 x 256 -- centered on (Cy, Cx)
        Cy = int(Cy)
        Cx = int(Cx)
        # cropped_gray = scaled_gray[Cy-128:Cy+128, Cx-128:Cx+128]
        # cropped_gray = scaled_gray[Cy-144-15:Cy+144-15, Cx-128:Cx+128]
        add_vertical = IMAGE_HEIGHT - cropped_image.shape[0]
        add_horizontal = IMAGE_WIDTH - cropped_image.shape[1]
        # if (add_vertical > 0) or (add_horizontal > 0):
        #     pad_tuple = ((floor(add_vertical/2), ceil(add_vertical/2)), (floor(add_horizontal/2), ceil(add_horizontal/2)))
        #     cropped_gray = np.pad(scaled_gray, pad_tuple, 'constant', constant_values=((0,0),(0,0)))
        # else:
        #     cropped_gray = scaled_gray

        # cropped_image = scaled_image

        if cropped_image.ndim == 2:
            if (add_vertical > 0):
                pad = ((floor(add_vertical/2), ceil(add_vertical/2)), (0, 0))
                cropped_image = np.pad(cropped_image, pad,
                                      'constant', constant_values=((0, 0), (0, 0)))
            elif (add_vertical < 0):
                pre_clip = floor(abs(add_vertical)/2)
                pos_clip = ceil(abs(add_vertical)/2)
                cropped_image = cropped_image[pre_clip: -pos_clip, :]
            if (add_horizontal > 0):
                pad = ((0, 0), (floor(add_horizontal/2), ceil(add_horizontal/2)))
                cropped_image = np.pad(cropped_image, pad,
                                      'constant', constant_values=((0, 0), (0, 0)))
            elif (add_horizontal < 0):
                pre_clip = floor(abs(add_horizontal)/2)
                pos_clip = ceil(abs(add_horizontal)/2)
                cropped_image = cropped_image[:, pre_clip: -pos_clip]
        else:  # new lines to account for third channel of an RGB image
            if (add_vertical > 0):
                pad = ((floor(add_vertical/2), ceil(add_vertical/2)), (0, 0), (0, 0))
                cropped_image = np.pad(cropped_image, pad,
                                      'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            elif (add_vertical < 0):
                pre_clip = floor(abs(add_vertical)/2)
                pos_clip = ceil(abs(add_vertical)/2)
                cropped_image = cropped_image[pre_clip: -pos_clip, :, :]
            if (add_horizontal > 0):
                pad = ((0, 0), (floor(add_horizontal/2), ceil(add_horizontal/2)), (0, 0))
                cropped_image = np.pad(cropped_image, pad,
                                      'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            elif (add_horizontal < 0):
                pre_clip = floor(abs(add_horizontal)/2)
                pos_clip = ceil(abs(add_horizontal)/2)
                cropped_image = cropped_image[:, pre_clip: -pos_clip, :]

        rows, cols = cropped_image.shape[0], cropped_image.shape[1]
        Cy, Cx = rows/2, cols/2

        # Rescale intensity to [0 255]
        final_image = exposure.rescale_intensity(cropped_image)

        DirBits = outfile.split(os.path.sep)
        go_back = -len(DirBits[-1])
        save_to_dir = outfile[0:go_back]
        save_to_this = DirBits[-1]

        if color_of_result == 'rgb':
            # aligned_gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
            # final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
            aligned_gray = np.array(PilImage.fromarray(final_image).convert("L"))
            final_image = np.array(PilImage.fromarray(final_image))
        else:
            aligned_gray = final_image

        os.chdir(save_to_dir)
        imsave(save_to_this, final_image)
        os.chdir(current_working_dir)
    print("\n\n------------------------------------------------------------\n")
    print("\nThis directory:\n")
    print("\t" + output_dir + "\n")
    print("should now be populated with aligned faces.\n")
    print("Please examine the contents.\n\n")
    print("\n--------------------------------------------------------------\n")
    # Write a specs.csv file to record constant image dimensions
    fieldnames = ["adjust_size", "image_height", "image_width", "eye_distance"]
    with open(output_dir + "specs.csv", mode="w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"adjust_size": adjust_size, "image_height":IMAGE_HEIGHT, "image_width":IMAGE_WIDTH,"eye_distance":EYE_DISTANCE})
    # return mean_shape, new_shapes, source_files
    return output_dir


def place_aperture(source_dir, file_prefix='', file_postfix='jpg',
                   aperture_type="MossEgg", no_save=False, contrast_norm="max",
                   color_of_result='grayscale'):
    if color_of_result == 'rgb':
        assert (contrast_norm=="max") or (contrast_norm==None)

    # files, output_dir = make_files(source_dir, file_prefix, file_postfix,
    #                                new_dir="windowed")
    # source_files = files[0]
    # output_files = files[1]

    output_dir = clone_directory_tree(source_dir, new_dir="windowed",
                                      FilePrefix=file_prefix,
                                      FilePostfix=file_postfix)

    landmark_features, files = get_landmark_features(source_dir, output_dir)
    source_files = files[0]
    output_files = files[1]

    # if len((set(source_files) - set(source_files_))) !=0:
    #     print("source_files returned by make_files() and get_landmark_features() are different.")
    #     print(source_files[0] + "\n" + source_files_[0])
    #     return

    shapes = np.array(landmark_features['AllSubjectsLandmarksDict'])
    # source_files = files[0]
    FilePostfix = source_files[0].split("/")[-1].split(".")[-1]

    # Points for all faces, mean face, and center of all landmarks.
    mean_shape = shapes.mean(axis=0)
    MX, MY = mean_shape[0::2], mean_shape[1::2]
    CX, CY = MX.mean(), MY.mean()
    # size = cv2.imread(source_files[0]).shape[0:2]
    size = np.array(PilImage.open(source_files[0]).convert("L")).shape[0:2]

    aperture_good = True
    if aperture_type == "Ellipse":
        X = shapes[:, 0::2].reshape(-1,)
        Y = shapes[:, 1::2].reshape(-1,)

        # Longest vertical length of ellipse that fits within image.
        if (size[0] / 2) < CY:
            ellipse_height = (size[0] - CY) * 2
        elif (size[0] / 2) > CY:
            ellipse_height = CY * 2
        else:
            ellipse_height = size[0]
        semi_major = ellipse_height / 2

        semi_minor = fit_ellipse_semi_minor(semi_major=semi_major,
                                            landmarks=(X, Y),
                                            center=(CX, CY))

        the_aperture = make_ellipse_map(semi_minor, semi_major,
                                        (CX, CY), size, soften=True)
    elif aperture_type == "MossEgg":
        the_aperture = make_moss_egg(landmark_features, (CX, CY),
                                     size, fraction_width=47/100, soften=True)[0]
    else:
        print("Error: aperture_type should be Ellipse or MossEgg.")
        aperture_good = False
        # the_aperture = []
    if no_save:
        return the_aperture
    if aperture_good:
        # # Need this to create all subfolders in ./in_aperture
        # files = make_files(source_dir, FilePrefix="",
        #                    FilePostfix=FilePostfix, new_dir="windowed")
        # output_files = files[1]
        if color_of_result=='grayscale':
            for infile, outfile in zip(source_files, output_files):
                # image = cv2.imread(infile)
                # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = np.array(PilImage.open(infile).convert("L"))
                gray = contrast_stretch(gray, inner_locs=(the_aperture * 255) > 16,
                                        type=contrast_norm)
                BGRA = make_four_channel_image(img=gray, aperture=the_aperture)

                out_png = outfile.split(".")[0] + ".png"
                print(outfile)
                print(out_png)
                # cv2.imwrite(out_png, BGRA)
                PilImage.fromarray(BGRA).save(out_png)
        elif color_of_result=='rgb':
            for infile, outfile in zip(source_files, output_files):
                # rgb = cv2.imread(infile)
                rgb = np.array(PilImage.open(infile))
                rgb = contrast_stretch(rgb, inner_locs=[], type=contrast_norm)
                BGRA = make_four_channel_image(img=rgb, aperture=the_aperture)

                out_png = outfile.split(".")[0] + ".png"
                print(outfile)
                print(out_png)
                # cv2.imwrite(out_png, BGRA)
                PilImage.fromarray(BGRA).save(out_png)
        print("\n\n--------------------------------------------------------\n")
        print("\nThere should now be a bunch of faces set in apertures in:\n")
        print("\t" + output_dir + "\n")
        print("Please examine the contents.\n\n")
        print("\n----------------------------------------------------------\n")
        return the_aperture, output_dir


def get_mean_image(file_path, max_inner_face_contrast=False):
    with io.open(file_path + '/landmarks.txt', 'r') as f:
        imported_landmarks = json.loads(f.readlines()[0].strip())
    guys = list(imported_landmarks.keys())
    gray_0 = np.array(PilImage.open(file_path + guys[0]).convert("L"))
    shape_0 = gray_0.shape

    if max_inner_face_contrast:
        # Normalize contrast within aperture, common mean of 127.5
        the_aperture = place_aperture(file_path, file_path, no_save=True)
        inner_map = (the_aperture * 255) > 16
        gray_0 = contrast_stretch(gray_0, inner_locs=inner_map, type="mean_127") - 127.5
    else:
        # UINT8 to double. Center and normalize
        gray_0 = gray_0.astype(float)
        gray_0 -= gray_0.mean()
        gray_0 = gray_0 / gray_0.std()

    mean_image = gray_0
    for guy in guys[1:]:
        gray = np.array(PilImage.open(file_path + guy).convert("L"))
        if gray.shape==shape_0:

            if max_inner_face_contrast:
                # Normalize contrast within aperture, common mean of 127.5
                gray = contrast_stretch(gray, inner_locs=inner_map, type="mean_127") - 127.5
            else:
                # UINT8 to double. Center and normalize
                gray = gray.astype(float)
                gray -= gray.mean()
                gray = gray / gray.std()

            mean_image += gray
        else:
            print("_get_mean_image() requires that all images are same dimensions!!")
            mean_image = None
            return mean_image
    # print("Go back to [0-255]")
    mean_image = mean_image / len(guys)
    return mean_image


def warp_to_mean_landmarks(source_dir, file_prefix='', file_postfix='jpg'):

    output_dir = clone_directory_tree(source_dir, new_dir="warp-to-mean",
                                      FilePrefix=file_prefix,
                                      FilePostfix=file_postfix)

    landmark_features, files = get_landmark_features(source_dir, output_dir,
                                                     exclude_features=['jawline',
                                                     'left_iris', 'right_iris'])
    source_files = files[0]
    L = np.array(landmark_features['AllSubjectsLandmarksDict'])
    # source_files = files[0]

    # Image size and number of images
    # size = cv2.imread(source_files[0]).shape[0:2]
    size = np.array(PilImage.open(source_files[0]).convert("L")).shape[0:2]
    num_faces = len(source_files)

    # Points for all faces, mean face, and center of all landmarks.
    ML = L.mean(axis=0)
    bx = ML[::2]
    by = ML[1::2]
    base = (bx, by)

    # Important -> imarray and target (landmarks)
    original_images = np.zeros((num_faces, size[0], size[1]))
    for i, of in enumerate(source_files):
        # image = cv2.imread(of)
        # original_images[i, :, :] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        original_images[i, :, :] = np.array(PilImage.open(of).convert("L"))
        tx = L[i, ::2]
        ty = L[i, 1::2]
        target = (tx, ty)

    base = np.transpose(np.array(base))

    warped_to_mean = np.zeros((num_faces, size[0], size[1]))
    for i, of in enumerate(source_files):
        im = original_images[i, :, :]
        im -= im.min()
        im = im / im.max()

        tx = L[i, ::2]
        ty = L[i, 1::2]
        target = (tx, ty)
        target = np.transpose(np.array(target))

        warpim, tri, inpix, fwdwarpix = pawarp(im, base, target,
                                               interp='bilin')
        warped_to_mean[i, :, :] = warpim

    # Write to file
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, im in enumerate(warped_to_mean):
        sim = (im * 255).astype("uint8")
        out_png = output_dir + os.path.sep + "P" + str(i) + ".png"
        # cv2.imwrite(out_png, sim)
        PilImage.fromarray(sim).save(out_png)

    # Estimate landmarks.
    get_landmarks(output_dir, "P", "png")

    # Get aperture for inner-face
    the_aperture = place_aperture(output_dir, output_dir, no_save=True)
    inner_map = (the_aperture * 255) > 16

    # Normalize contrast within aperture, common mean of 127.5
    WTMN = np.zeros((num_faces, size[0], size[1]))
    for i, im in enumerate(warped_to_mean):
        nim = contrast_stretch(im, inner_locs=inner_map, type="mean_127")
        out_png = output_dir + os.path.sep + "N" + str(i) + ".png"
        # cv2.imwrite(out_png, nim)
        PilImage.fromarray(nim).save(out_png)
        WTMN[i, :, :] = nim

    # Mean of warped and normalized faces, and save to file
    AverageOfMorphs = contrast_stretch(WTMN.mean(axis=0), type="max")
    out_png = output_dir + os.path.sep + "N-mean.png"
    # cv2.imwrite(out_png, AverageOfMorphs)
    PilImage.fromarray(AverageOfMorphs).save(out_png)

    return original_images, WTMN


def morph_between_two_faces(source_dir, do_these, num_morphs, file_prefix='',
                            file_postfix='jpg', new_dir = "morphed",
                            weight_texture=True):
    assert len(do_these) == 2

    output_dir = clone_directory_tree(source_dir, new_dir=new_dir,
                                      FilePrefix=file_prefix,
                                      FilePostfix=file_postfix)

    landmark_features, files = get_landmark_features(source_dir, output_dir)
    lms = np.array(landmark_features['AllSubjectsLandmarksDict'])
    SF = files[0]

    # select 2 faces in list do_these
    lms = lms[do_these, :]
    source_files = [SF[i] for i in do_these]

    # Image size and number of images
    # size = cv2.imread(source_files[0]).shape[0:2]
    size = np.array(PilImage.open(source_files[0]).convert("L")).shape[0:2]
    num_faces = 2

    # # Points for all faces, mean face, and center of all landmarks.
    # ML = lms.mean(axis=0)
    # bx = ML[::2]
    # by = ML[1::2]
    # base = (bx, by)

    original_images = np.zeros((num_faces, size[0], size[1]))
    F, L = [], []
    for i, of in enumerate(source_files):
        # image = cv2.imread(of)
        # im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im = np.array(PilImage.open(of).convert("L"))
        im -= im.min()
        F.append(im / im.max())

        tx = lms[i, ::2]
        ty = lms[i, 1::2]
        target = (tx, ty)
        L.append(np.transpose(np.array(target)))  # point by [x,y]

    p = [i/(num_morphs-1) for i in range(num_morphs)]
    face_array = np.zeros((num_morphs, F[0].shape[0], F[0].shape[1]))
    for i in range(num_morphs):
        N = p[i]*L[0] + (1-p[i])*L[1]
        F0, tri, inpix, fwdwarpix = pawarp(F[0], N, L[0], interp='bilin')
        F1, tri, inpix, fwdwarpix = pawarp(F[1], N, L[1], interp='bilin')
        if weight_texture:
            M = p[i]*F0 + (1-p[i])*F1
        else:
            M = (F0 + F1) / 2
        face_array[i, :, :] = M

    # From face A to B
    face_array =np.flip(face_array, axis=0)

    # Write to morphed faces to file
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, im in enumerate(face_array):
        sim = (im * 255).astype("uint8")
        out_png = output_dir + os.path.sep + "P" + str(i) + ".png"
        # cv2.imwrite(out_png, sim)
        PilImage.fromarray(sim).save(out_png)

    # Estimate landmarks.
    get_landmarks(output_dir, "P", "png")

    # Get aperture for inner-face
    the_aperture = place_aperture(output_dir, no_save=True)
    inner_map = (the_aperture * 255) > 16

    # Normalize contrast within aperture, common mean of 127.5
    WTMN = np.zeros((num_morphs, size[0], size[1]))
    for i, im in enumerate(face_array):
        im[im>1] = 1
        im[im<0] = 0
        nim = contrast_stretch(im, inner_locs=inner_map, type="mean_127")
        out_png = output_dir + os.path.sep + "N" + str(i) + ".png"
        # cv2.imwrite(out_png, nim)
        PilImage.fromarray(nim).save(out_png)
        WTMN[i, :, :] = nim

    return WTMN, p, output_dir
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
