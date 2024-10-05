Automatic Face Alignment (AFA)
================
Carl M. Gaspar & Oliver G.B. Garrod

#### You have lots of photos of faces like this:
![](demos/demo_1_alignment/collage_originals.png)

#### But you want to line up all of the faces like this:
![](demos/demo_1_alignment/collage_aligned.png)

#### Perhaps you would also like to window the faces to show only inner facial features like this:
![](demos/demo_1_alignment/collage_aligned_windowed.png)

#### All of the above can be done using AFA like this:
```python
import alignfaces as afa

faces_path = "/Users/Me/faces_for_my_study/"
afa.get_landmarks(faces_path)
aligned_path = afa.align_procrustes(faces_path)
afa.get_landmarks(aligned_path)
the_aperture, aperture_path = afa.place_aperture(aligned_path)
```
To better understand how to write a script for your specific purposes, we direct you to [demo 1](demos/demo_1_alignment/README.md). [Demo 1](demos/demo_1_alignment/README.md) also describes how AFA alignment works.

All of these functions depend on reliable detection of facial landmarks, which is provided by the [DLIB](http://dlib.net) library. Alignment is based on generalized Procrustes analysis (GPA), which extensively unit tested.

# Additional functions (warping)
Automatic landmark detection means that it is also easy to separate **shape** and **texture** in order to produce various kinds of **warped** images.

AFA provides functions for two types of face-warping manipulations common in face perception research.

### Morphing between faces
To learn how to do this please see [demo 2](demos/demo_2_morphing/README.md).

### Enhanced average of facial identity
To learn how to do this please see [demo 3](demos/demo_3_averaging/README.md).

# Setup

It is highly recommended that you have **conda** installed, preferably **miniconda** rather than full fat **anaconda**.

If you do have **conda**, then this is the easiest way to install:

```bash
conda create --name myenv conda-forge::dlib "python>=3.9" scikit-image

conda activate myenv

conda install -c conda-forge matplotlib
```

To install AFA next you have two options:

You either do this:

```bash
pip install "alignfaces @ git+https://git@github.com/SourCherries/auto-face-align.git"
```

Or if instead you want a readable and editable copy of AFA on your local machine, then first clone this repository, go to the root folder `auto-face-align`, and then do this:

```bash
pip install .
```

Regardless of how you installed AFA, the above process will create a new virtual environment called `myenv`. You can use another name for that. You'll need to activate this environment using `conda activate myenv` whenever you want to use AFA. To deactivate, simply type `conda deactivate myenv`.

If you have a readable/editable copy of AFA on your local machine, you will have copies of all the demos. Most users will want those demo scripts to get started on their projects.

Other users may want a readable/editable copy of AFA to contribute to AFA, or to evaluate AFA by running the analyses under `results` or the unit tests. To run the unit tests, go to the root folder `auto-face-align` then do this:

```bash
pip install -U pytest
pytest -v src/alignfaces/tests/
```

# How well does this work?
In addition to unit-testing critical computations, I evaluated both landmark estimation (DLIB) and the outcome of the entire alignment procedure using various face databases. The results are described [here](results/README.md).

<!-- ## Ensure that you have the proper C compiler
On Linux, you will already have an appropriate C compiler.

On Windows, you need to install Microsoft Visual Studio.

On Mac, you need to install Xcode Command Line Tools.
1. Find an Xcode version compatible with your [macOS version](https://en.wikipedia.org/wiki/Xcode).
2. Get the right version of [Xcode Command Line Tools](https://developer.apple.com/downloads/index.action).
``` -->

# Citation
If you use this package for your research, please cite the following preprint:
>Gaspar, C. M., & Garrod, O. G. B. (2021, November 8). A Python toolbox for Automatic Face Alignment (AFA). Retrieved from psyarxiv.com/erc8a

DOI:
>10.31234/osf.io/erc8a

# License
This module is under an Apache-2.0 license.
