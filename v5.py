# run:
# python3 v5.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

from djitellopy import Tello
import cv2
import pygame
from pygame.locals import *
import numpy as np
import time
from imutils.video import VideoStream
import argparse
import imutils
import sys

# adding and parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# change in speed in follow mode
S = 18
W = 8
M = 20
Q = 10
P = 25
# change in speed in override mode
N = 60
# fps of pygame display
FPS = 25


class FrontEnd(object):

    def __init__(self):
        # initiate pygame
        pygame.init()

        self.screen = pygame.display.set_mode([960, 720])

        # initiate Tello methods to interact with the drone
        self.tello = Tello()

        # drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10
        self.send_rc_control = False

        # create timer
        pygame.time.set_timer(USEREVENT + 1, 50)

    def run(self):

        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

            # reads video stream from Tello
        frame_read = self.tello.get_frame_read()
        time.sleep(2.0)
        should_stop = False
        override = True
        # what happens when the stream is on
        while not should_stop:
            for event in pygame.event.get():
                # updating
                if event.type == USEREVENT + 1:
                    self.update()
                # stream turns off when pygame window is closed
                elif event.type == QUIT:
                    should_stop = True
                # if a key is physically pressed
                elif event.type == KEYDOWN:
                    # if the escape is pressed, stream turns off
                    if event.key == K_ESCAPE:
                        should_stop = True
                    elif event.key == K_SPACE:
                        override = True
                    elif event.key == K_RETURN:
                        override = False
                    if override == True:
                        self.keydown(event.key)
                elif event.type == KEYUP:
                    self.keyup(event.key)
            # if can't read video feed (e.g. tello stops transmitting)
            if frame_read.stopped:
                frame_read.stop()
                break
            # fill up colors/window
            self.screen.fill([0, 0, 0])
            frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = np.flipud(frame)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))

            # telemtry values
            batt = self.tello.get_battery()
            flytime = self.tello.get_flight_time()

            # adding facial recognition bounding box
            cam = frame_read.frame
            (h, w) = cam.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(cam, (300, 300)), 1.0, (300, 300),
                                         (104.0, 177.0, 123.0))
            # initiate facial recognition model
            net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
            # passing blob through neural network to detect faces
            net.setInput(blob)
            detections = net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < args["confidence"]:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (Xi, Yi, Xf, Yf) = box.astype("int")
                cv2.rectangle(cam, (Xi, Yi), (Xf, Yf), (255, 255, 255), 2)

                # center of screen
                screen_center_x = 480
                screen_center_y = 360
                # draw center of screen
                cv2.circle(cam, (screen_center_x, screen_center_y), 6, (0, 0, 255), 2, 8, 0)

                # target area
                TXi = screen_center_x - 150
                TXf = screen_center_x + 150
                TYi = screen_center_y - 150
                TYf = screen_center_y + 150
                # draw target area
                cv2.rectangle(cam, (TXi, TYi), (TXf, TYf), (255, 255, 0), 2)

                # center of bounding box
                box_center_x = (Xf + Xi) / 2
                box_center_y = (Yf + Yi) / 2
                # draw the center of bounding box
                cv2.circle(cam, (int(box_center_x), int(box_center_y)), 6, (255, 0, 0), 2, 8, 0)

                # area of bounding box
                box_area = (Xf - Xi) * (Yf - Yi)
                # 65000 ~ 75000

                if override == False:
                    if box_center_x > TXf:
                        self.yaw_velocity = P
                        self.left_right_velocity = W
                    elif box_center_x < TXi:
                        self.yaw_velocity = -P
                        self.left_right_velocity = -W
                    else:
                        self.yaw_velocity = 0
                        self.left_right_velocity = 0

                    if box_center_y > TYf:
                        self.up_down_velocity = -S
                    elif box_center_y < TYi:
                        self.up_down_velocity = S
                    else:
                        self.up_down_velocity = 0

                    if box_area > 40000:
                        self.for_back_velocity = -M
                    elif box_area > 30000 and box_area < 40000:
                        self.for_back_velocity = -Q
                    elif box_area > 10000 and box_area < 20000:
                        self.for_back_velocity = Q
                    elif box_area < 10000:
                        self.for_back_velocity = M
                    else:
                        self.for_back_velocity = 0

            # show telementry data
            cv2.putText(cam, "Battery %: " + str(batt), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)
            cv2.putText(cam, "Flight time: " + str(flytime), (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)
            cv2.putText(cam, "Manual mode? " + str(override), (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)
            cv2.imshow("Display Window", cam)

            # frequency of update
            time.sleep(1 / FPS)
        # return (allocated memory) to the store of available RAM
        self.tello.end()

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = N
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -N
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -N
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = N
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = N
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -N
        elif key == pygame.K_a:  # set yaw clockwise velocity
            self.yaw_velocity = -N
        elif key == pygame.K_d:  # set yaw counter clockwise velocity
            self.yaw_velocity = N

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                                       self.up_down_velocity,
                                       self.yaw_velocity)


def main():
    frontend = FrontEnd()
    frontend.run()


if __name__ == '__main__':
    main()
