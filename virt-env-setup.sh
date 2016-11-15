#! /bin/bash

## Setting up virtual env

pushd .

# Install virtualenv
sudo pip install virtualenv

# Create virtualenv for BahmniOCR
mkdir ~/.virtualenvs
cd ~/.virtualenvs
virtualenv BahmniOCR

# Symlink OpenCV
cd ~/.virtualenvs/BahmniOCR/bin
ln -s /usr/local/lib/python2.7/site-packages/cv.py
ln -s /usr/local/lib/python2.7/site-packages/cv2.so

# Move to current directory
popd

# Setup framework python
cp frameworkpython ~/.virtualenvs/BahmniOCR/bin

# Activate virtual env
source ~/.virtualenvs/BahmniOCR/bin/activate

# Install dependencies
pip install -r requirements.txt

