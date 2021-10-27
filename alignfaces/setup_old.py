from setuptools import setup, find_packages

import sys
# import os

import pathlib

# The directory containing this file
# BASE_LOCATION = os.path.abspath(os.path.dirname(__file__))
# BASE_LOCATION = pathlib.Path(__file__).parent
BASE_LOCATION = pathlib.Path.cwd()
VERSION_FILE = 'VERSION'
REQUIRES_FILE = 'REQUIREMENTS'

# The text of the README file
README = (BASE_LOCATION.parent / "README.md").read_text()

def filter_comments(fd):
    return filter(lambda l: l.strip().startswith("#") is False, fd.readlines())


def readfile(filename, func):
    try:
        # with open(os.path.join(BASE_LOCATION, filename)) as f:
        with open(BASE_LOCATION / filename) as f:
            data = func(f)
    except (IOError, IndexError):
        sys.stderr.write(u"""
Can't find '%s' file. This doesn't seem to be a valid release.
If you are working from a git clone, run:
    make describe
    setup.py develop
To build a valid release, run:
    make all
""" % filename)
        sys.exit(1)
    return data


def get_version():
    return readfile(VERSION_FILE, lambda f: f.read().strip())


def get_requires():
    return readfile(REQUIRES_FILE, filter_comments)


setup(
    name="alignfaces",
    description="Automatically align and warp face images",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/SourCherries/facepackage-slim",
    author="Carl Michael Gaspar",
    author_email="carl.michael.gaspar@icloud.com",
    package_dir={'': 'src'},
    packages=find_packages('src'),
    package_data={'src/alignfaces/data/':
                  ['shape_predictor_68_face_landmarks.dat']},
    version=get_version(),
    install_requires=get_requires(),
    include_package_data=True,
    zip_safe=False,
    test_suite="mypackage.tests")
