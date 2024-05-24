#!/usr/bin/env bash

# Author Carl's shell script to install and run demo on his macOS computers.
#   Already assumes that cmake and requirements.txt taken care of.

# Useful for debugging.

mother=$(pwd)

# Activate appropriate virtual environment.
status=$(uname -n)
if [ "$status" = "ML089328-10722" ]; then
  echo "This is Carl's ZU MacBook Pro (M1)."
  eval "$(conda shell.bash hook)"
  conda activate afa-test
elif [ "$status" = "Carls-iMac" ] ; then
  echo "This is Carl's iMac."
  source slimmest/bin/activate
elif [ "$status" = "Carls-MacBook-Pro.local" ] ; then
  echo "This is Carl's MacBook Pro."
  source activate F0
else
  echo "Unknown computer."
fi

# Install face toolbox.
if [ "$1" = "install" ] ; then
  #cd alignfaces
  python setup.py install
  #cd $mother
else
  echo "Type './quick_install_for_debugging.sh install' to install face toolbox."
  #cd alignfaces
  #cd $mother
fi

# Run unit tests.
cd "$mother/alignfaces/src/alignfaces/tests"
pytest -v
cd $mother


# Run basic sample script.
cd "$mother/demos/demo_1_alignment"
python run_demo.py

# Examine output of sample script.
find ./faces-aligned-windowed -name "*.png" -exec open {} \;


# In macOS at least, reverts to original directory and virtual enviroment
#   when this shell script terminates.

# END
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
