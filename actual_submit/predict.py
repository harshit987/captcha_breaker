import numpy as np
import cv2
from PIL import Image
from PIL import ImageFilter

import pyximport
pyximport.install()
import loop
from keras.models import load_model

# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a numpy array (not numpy matrix or scipy matrix) and a list of strings.
# Make sure that the length of the array and the list is the same as the number of filenames that
# were given. The evaluation code may give unexpected results if this convention is not followed.


def decaptcha(filenames):
    print("In Progress...")
    numChars,codes=loop.loop(filenames)
    return (numChars, codes)
