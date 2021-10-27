# from setuptools import setup, find_packages
import pathlib

# The directory containing this file
# BASE_LOCATION = os.path.abspath(os.path.dirname(__file__))

BASE_LOCATION = pathlib.Path(__file__).parent
pathlib.Path.cwd().parent / "README.md"

BASE_LOCATION = pathlib.Path.cwd()

# TEST_LOCATION = pathlib.Path(__file__)

# VERSION_FILE = 'VERSION'
# REQUIRES_FILE = 'REQUIREMENTS'

print(BASE_LOCATION)

# The text of the README file
README = (BASE_LOCATION.parent / "README.md").read_text()
