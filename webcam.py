from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import pygame
from pygame.locals import *

# constructing arguments and parsing arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
#                 help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
#                 help="path to Caffe pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#                 help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())
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
            cv2.imshow("Webcam", frame)
            cam = webcam.frame
            cv2.putText(cam, "body_detection: " + str(body_detection), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

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

            (height, width) = cam.shape[:2]



def main():
    frontend = FrontEnd()
    frontend.run()


if __name__ == '__main__':
    main()

# while True:
#     frame = webcam.read()
#     frame = imutils.resize(frame, width=400)
#
#     # convert frame into blob
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
#                                  (300, 300), (104.0, 177.0, 123.0))
#
#     net.setInput(blob)
#     detections = net.forward()
#
#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence < args["confidence"]:
#             continue
#         # compute bounding box
#         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#         (startX, startY, endX, endY) = box.astype("int")
#         # draw the bounding box around face
#         text = "{:.2f}%".format(confidence * 100)
#         y = startY - 10 if startY - 10 > 10 else startY + 10
#         cv2.rectangle(frame, (startX, startY), (endX, endY),
#                       (0, 0, 255), 2)
#         cv2.putText(frame, text, (startX, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
#
#         cv2.imshow("Frame", frame)
#         key = cv2.waitKey(1) & 0xFF
#
#         if key == ord("q"):
#             break
#
#     cv2.destroyAllWindows()
#     webcam.stop()
