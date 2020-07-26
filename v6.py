from djitellopy import Tello
import cv2
import sys
import os
import time
import math
import numpy as np
from sys import platform
import argparse
import imutils
from imutils.video import VideoStream
import pygame
from pygame.locals import * 

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/python/openpose/Release')
os.environ['PATH'] = os.environ['PATH'] + ';' + \
  dir_path + '/Release;' + dir_path + '/bin;'

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
  params["num_gpu_start"] = 0
  params["disable_blending"] = False
  params["model_folder"] = dir_path + '/models'
  return params

FPS = 15

N = 30

class Fly(object):
  def __init__(self):
    pygame.init()
    self.screen = pygame.display.set_mode([256, 144])
    self.tello = Tello()

    # drone velocities
    self.for_back_velocity = 0
    self.left_right_velocity = 0
    self.up_down_velocity = 0
    self.yaw_velocity = 0
    self.speed = 10
    self.send_rc_control = False

    pygame.time.set_timer(USEREVENT + 1, 50)

  def run(self):  
    # setup 
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

    # setup params for openpose
    params = set_params()
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    
    frame_read = self.tello.get_frame_read()
    time.sleep(2.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    should_stop = False
    override = True
    while not should_stop:
      # keyboard interactions
      for event in pygame.event.get():
        # updating 
        if event.type == USEREVENT + 1:
          self.update()
        elif event.type == QUIT:
          should_stop = True
        elif event.type == KEYDOWN:
          if event.key == K_ESCAPE:
            should_stop = True
          # if override == True:
          self.keydown(event.key)
        elif event.type == KEYUP:
          self.keyup(event.key)

      if frame_read.stopped:
        frame_read.stop()
        break

      # telementry values

      # flytime = self.tello.get_flight_time()
    
      # process frame read from tello drone

      cam = frame_read.frame
      cam = cv2.resize(cam, (426, 240))
      datum = op.Datum()
      datum.cvInputData = cam
      opWrapper.emplaceAndPop([datum])

      if (isinstance(datum.poseKeypoints, np.ndarray) 
            and datum.poseKeypoints.size == 75):
        noseX = datum.poseKeypoints[0][0][0]
        noseY = datum.poseKeypoints[0][0][1]
        cv2.circle(cam, (noseX, noseY), 5, (0, 255, 255))
      
      batt = self.tello.get_battery()
      cv2.putText(cam, "Battery %: " + str(batt), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (255, 255, 255), 2)

      cv2.imshow('Good Tello', cam)
      time.sleep(1 / FPS)
    self.tello.end()

  def keydown(self, key):
    if key == pygame.K_UP:  # set forward velocity
      print(self.send_rc_control)
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
    if self.send_rc_control:
      self.tello.send_rc_control(self.left_right_velocity, 
        self.for_back_velocity, self.up_down_velocity, self.yaw_velocity)

def main():
  fly = Fly()
  fly.run()

if __name__ == '__main__':
  main()
