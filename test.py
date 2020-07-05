import cv2
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/python/openpose/Release')
os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/Release;' + dir_path + '/bin;'

import pyopenpose as openpose

print("hello")