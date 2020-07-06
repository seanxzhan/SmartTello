from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import pygame
from pygame.locals import *
import os

# constructing arguments and parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
#
# print("[INFO] loading model...")
# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
#
# # turning on webcam
# print("[INFO] starting video stream...")
webcam = VideoStream(src=0).start()
time.sleep(2.0)


class FrontEnd(object):

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode([960, 720])

    def run(self):
        showing = True
        body_detection = False

        while showing:
            frame = webcam.read()
            # cv2.imshow("Webcam", frame)
            cam = webcam.frame
            cv2.putText(cam, "body_detection: " + str(body_detection), 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        showing = False
                    elif event.key == K_b:
                        if body_detection:
                            body_detection = False
                        else:
                            body_detection = True
            if not showing:
                break

            npoints = 15
            POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], 
                [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]

            net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

            # convert each frame into a blob
            (height, width) = cam.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(cam, (300, 300)), 
                1.0 / 255, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            output = net.forward()
            
            H = output.shape[2]
            W = output.shape[3]
            points = []
            threshold = 0.1

            for i in range(npoints):
                probMap = output[0, i, :, :]
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                x = (width * point[0]) / W
                y = (height * point[1]) / H

                if prob > threshold :
                    cv2.circle(cam, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.putText(cam, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)
                    points.append((int(x), int(y)))
                else: 
                    points.append(None)

            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]

                if points[partA] and points[partB]:
                    cv2.line(cam, points[partA], points[partB], (0, 255, 255), 2)
                    cv2.circle(cam, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            
            cv2.imshow("Output", cam)


def main():
    frontend = FrontEnd()
    frontend.run()


if __name__ == '__main__':
    main()
