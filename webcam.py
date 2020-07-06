import cv2
import sys
import os
import time
import math
import numpy as py
from sys import platform
import argparse
from imutils.video import VideoStream

sys.path.append('C:\\Users\\seanz\\CS_Projects\\openpose\\build\\python\\openpose\\Release')
os.environ['PATH'] = os.environ['PATH'] + ';' +\
    'C:\\Users\\seanz\\CS_Projects\\openpose\\build\\x64\\Release' + ';' +\
    'C:\\Users\\seanz\\CS_Projects\\openpose\\build\\bin'
print(sys.path)

import pyopenpose as op

def set_params():
        params = dict()
        params["logging_level"] = 3
        params["output_resolution"] = "-1x-1"
        params["net_resolution"] = "-1x368"
        params["model_pose"] = "BODY_25"
        params["alpha_pose"] = 0.6
        params["scale_gap"] = 0.3
        params["scale_number"] = 1
        params["render_threshold"] = 0.05
        # If GPU version is built, and multiple GPUs are available, set the ID here
        params["num_gpu_start"] = 0
        params["disable_blending"] = False
        # Ensure you point to the correct path where models are located
        params["model_folder"] = 'C:\\Users\\seanz\\CS_Projects\\openpose\\models'
        return params

def main():
        params = set_params()

        #Constructing OpenPose object allocates GPU memory
        # openpose = OpenPose(params)
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        #Opening OpenCV stream
        webcam = VideoStream(src=0).start()

        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:
                
                # ret, img = stream.read()
                frame = webcam.read()

                datum = op.Datum()
                # imageToProcess = cv2.imread(frame)
                datum.cvInputData = frame
                opWrapper.emplaceAndPop([datum])

                # Display the stream
                cv2.putText(frame,'OpenPose using Python-OpenCV',(20,30), font, 1,(255,255,255),1,cv2.LINE_AA)

                cv2.imshow('Human Pose Estimation', datum.cvOutputData)

                key = cv2.waitKey(1)

                if key==ord('q'):
                        break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


# trained model method - way to slow

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
#                 help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
#                 help="path to Caffe pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#                 help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())

# protoFile = "/pose_deploy_linevec_faster_4_stages.prototxt"
# weightsFile = "/pose_iter_160000.caffemodel"
# imageFile = "/genji.png"
# npoints = 15
# POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

# # net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# # this frame can be passed in from camera feed
# frame = cv2.imread(dir_path + imageFile)
# frameWidth = frame.shape[1]
# frameHeight = frame.shape[0]
# threshold = 0.1

# inWidth = 300
# inHeight = 300 

# blob = cv2.dnn.blobFromImage(
#     cv2.resize(frame, (300, 300)), 1.0 / 255, (inWidth, inHeight), 
#     (104.0, 177.0, 123.0), swapRB=False, crop=False)

# net.setInput(blob)
# output = net.forward()

# H = output.shape[2]
# W = output.shape[3]
# points = []

# for i in range(npoints):
#     probMap = output[0, i, :, :]
#     minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
#     x = (frameWidth * point[0]) / W
#     y = (frameHeight * point[1]) / H

#     if prob > threshold :
#         cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
#         cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)
#         points.append((int(x), int(y)))
#     else: 
#         points.append(None)

# for pair in POSE_PAIRS:
#     partA = pair[0]
#     partB = pair[1]

#     if points[partA] and points[partB]:
#         cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
#         cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

# cv2.imshow("Output", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()