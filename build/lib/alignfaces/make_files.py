from pathlib import Path
import os


# Run at outset of get_landmarks()
def get_source_files(MotherDir, FilePrefix="", FilePostfix="jpg"):
    if type(MotherDir) == str:
        MotherDir = Path(MotherDir)
    input_files = []
    for p in MotherDir.rglob(FilePrefix + "*" + FilePostfix):
        input_files.append(str(p))
        print(str(p))
    return input_files


# Run at outset of align_procrustes() and place_aperture()
def clone_directory_tree(MotherDir, new_dir="aligned", FilePrefix="", FilePostfix="jpg"):
    if type(MotherDir) == str:
        MotherDir = Path(MotherDir)    
    input_files = get_source_files(MotherDir, FilePrefix=FilePrefix, FilePostfix=FilePostfix)
    P = []
    for f in input_files:
        P.append(Path(f).parent)
    P = list(set(P))
    mothers_name = MotherDir.parts[-1]
    new_directory = MotherDir.parent / (str(mothers_name) + "-" + new_dir)
    print(new_directory)
    for p in P:
        new_dir = new_directory / p.relative_to(MotherDir)
        if not new_dir.exists():
            new_dir.mkdir(parents=False, exist_ok=False) # NEED TEST
        print(new_dir)
    return str(new_directory) + os.sep


# Within make_aligned_faces (align_procrustes and place_aperture),
#   get_landmark_features() is used to get only analyze those files with
#   landmarks and so returns a list of input and output filenames.
#   Where output_files determined by output_dir
# End
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
