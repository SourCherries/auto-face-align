import glob
import os

# Deprecated function (does too much!)
def make_files(MotherDir, FilePrefix="", FilePostfix="jpg", new_dir="aligned"):
    # Copy directory structure for output
    MotherBits = MotherDir.split(os.path.sep)
    go_back = -len(MotherBits[-2]) - 1
    GrannyDir = MotherDir[0:go_back]
    AuntieDir = GrannyDir + MotherBits[-2] + "-" + new_dir + os.path.sep

    if not os.path.isdir(AuntieDir):
        os.mkdir(AuntieDir)

    input_files = []
    output_files = []
    for infile in glob.iglob(MotherDir + "**" + os.path.sep +
                             FilePrefix + "*." + FilePostfix, recursive=True):
        input_files.append(infile)
        GiveToAuntie = infile[len(MotherDir):]
        output_files.append(AuntieDir + GiveToAuntie)
        AdoptedBits = GiveToAuntie.split(os.path.sep)
        AdoptedBits = AdoptedBits[:-1]
        CheckForThis = AuntieDir
        for next_dir in AdoptedBits:
            CheckForThis = CheckForThis + next_dir + os.path.sep
            if not os.path.isdir(CheckForThis):
                os.mkdir(CheckForThis)

    assert len(input_files) == len(output_files)
    # # Write lists to files.
    # with open(GrannyDir + 'input_files.data', 'wb') as filehandle:
    #     pickle.dump(input_files, filehandle)
    # with open(GrannyDir + 'output_files.data', 'wb') as filehandle:
    #     pickle.dump(output_files, filehandle)
    files = []
    files.append(input_files)
    files.append(output_files)
    output_dir = AuntieDir
    return files, output_dir


# Run at outset of get_landmarks()
def get_source_files(MotherDir, FilePrefix="", FilePostfix="jpg"):
    MotherBits = MotherDir.split(os.path.sep)
    input_files = []
    for infile in glob.iglob(MotherDir + "**" + os.path.sep +
                             FilePrefix + "*." + FilePostfix, recursive=True):
        input_files.append(infile)
    return input_files


# Run at outset of align_procrustes() and place_aperture()
def clone_directory_tree(MotherDir, new_dir="aligned", FilePrefix="", FilePostfix="jpg"):
    MotherBits = MotherDir.split(os.path.sep)
    go_back = -len(MotherBits[-2]) - 1
    GrannyDir = MotherDir[0:go_back]
    AuntieDir = GrannyDir + MotherBits[-2] + "-" + new_dir + os.path.sep

    if not os.path.isdir(AuntieDir):
        os.mkdir(AuntieDir)

    for infile in glob.iglob(MotherDir + "**" + os.path.sep +
                             FilePrefix + "*." + FilePostfix, recursive=True):

        GiveToAuntie = infile[len(MotherDir):]

        AdoptedBits = GiveToAuntie.split(os.path.sep)
        AdoptedBits = AdoptedBits[:-1]
        CheckForThis = AuntieDir
        for next_dir in AdoptedBits:
            CheckForThis = CheckForThis + next_dir + os.path.sep
            if not os.path.isdir(CheckForThis):
                os.mkdir(CheckForThis)

    output_dir = AuntieDir
    return output_dir


# Within make_aligned_faces (align_procrustes and place_aperture),
#   get_landmark_features() is used to get only analyze those files with
#   landmarks and so returns a list of input and output filenames.
#   Where output_files determined by output_dir
# End
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
