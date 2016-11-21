#! /bin/bash

# Setting up a virtual env for BahmniOCR

PROJECT_DIR=$(pwd)
VIRT_BIN=$HOME/.virtualenvs/BahmniOCR/bin

# Install virtualenv
pip install --user virtualenv

# TODO The following line works on macOS only.
export PATH=$PATH:$HOME/Library/Python/2.7/bin

# Create virtualenv for BahmniOCR
mkdir $HOME/.virtualenvs
cd $HOME/.virtualenvs
virtualenv BahmniOCR

cd $VIRT_BIN

# Activate virtual env
source activate

# Install dependencies
pip install -r $PROJECT_DIR/requirements.txt

# Setup framework python
cp $PROJECT_DIR/frameworkpython $VIRT_BIN
mv python python.bak
mv frameworkpython python

# Symlink OpenCV
cd $HOME/.virtualenvs/BahmniOCR/lib/python2.7/site-packages
ln -s /usr/local/lib/python2.7/site-packages/cv.py
ln -s /usr/local/lib/python2.7/site-packages/cv2.so

