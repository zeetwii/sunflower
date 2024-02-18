import cv2 # needed for open CV
import numpy as np # needed for contour math
import math # needed for floor
from scipy.spatial import distance # need to calc distance from center


import os # needed for IMU
import sys # needed for IMU
import time # needed for IMU
import smbus # needed for IMU

from imusensor.MPU9250 import MPU9250 # needed for IMU
from imusensor.filters import kalman # needed for IMU

# set up IMU
address = 0x68
bus = smbus.SMBus(1)
imu = MPU9250.MPU9250(bus, address)
imu.begin()

sensorfusion = kalman.Kalman()

imu.readSensor()
imu.computeOrientation()
sensorfusion.roll = imu.roll
sensorfusion.pitch = imu.pitch
sensorfusion.yaw = imu.yaw

count = 0
currTime = time.time()

# do one run to get defaults
imu.readSensor()
imu.computeOrientation()
newTime = time.time()
dt = newTime - currTime
currTime = newTime
sensorfusion.computeAndUpdateRollPitchYaw(imu.AccelVals[0], imu.AccelVals[1], imu.AccelVals[2], imu.GyroVals[0], imu.GyroVals[1], imu.GyroVals[2],\
												imu.MagVals[0], imu.MagVals[1], imu.MagVals[2], dt)
defaultRoll = sensorfusion.roll
defaultPitch = sensorfusion.pitch
defaultYaw = sensorfusion.yaw


cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('test.mp4')
#cap = cv2.VideoCapture('m2-res_480p.mp4')
#cap = cv2.VideoCapture('9zYxpT0f6jW2IVB5.mp4')

while True:
    
    imu.readSensor()
    imu.computeOrientation()
    newTime = time.time()
    dt = newTime - currTime
    currTime = newTime

    sensorfusion.computeAndUpdateRollPitchYaw(imu.AccelVals[0], imu.AccelVals[1], imu.AccelVals[2], imu.GyroVals[0], imu.GyroVals[1], imu.GyroVals[2],\
												imu.MagVals[0], imu.MagVals[1], imu.MagVals[2], dt)
                                                
    roll = defaultRoll - sensorfusion.roll
    pitch = defaultPitch - sensorfusion.pitch
    yaw = defaultYaw - sensorfusion.yaw
    
    if roll > 100:
        roll == 100
    elif roll < -100:
        roll == -100
    else:
        roll = round(roll)
        
    if pitch > 100:
        pitch == 100
    elif pitch < -100:
        pitch == -100
    else:
        pitch = round(pitch) * 10
        
    if yaw > 100:
        yaw == 100
    elif yaw < -100:
        yaw == -100
    else:
        yaw = round(yaw) * 10

    #print(f"ROLL: {str(roll)} , PITCH: {str(pitch)} , YAW: {str(yaw)}")

    
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 10, 70)
    ret, mask = cv2.threshold(canny, 70, 255, cv2.THRESH_BINARY)
    
    # show the canny filter
    cv2.imshow('Video feed', mask)
    
    #find center of image and draw it (green circle)
    (h, w) = img.shape[:2]
    #print(f"{str(h)} {str(w)}")
    imageCenter = [w//2, h//2]
    
    modifiedW = w//2 + yaw
    modifiedH = h//2 - pitch
    
    directionCenter = [modifiedW, modifiedH]
    
    cv2.circle(img, (modifiedW, modifiedH), 7, (0, 255, 0), -1)
    
    # find the contours in the edged image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # draw the contours to check
    #cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    
    # sorts from largest to smallest
    sortedContours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # find the top 10% largest contours
    if len(sortedContours) > 1:
        #print(f"Number of contours: {str(len(sortedContours))}")
        
        topContours = sortedContours[0 : math.floor(len(sortedContours) * 0.25)]
        #cv2.drawContours(img, topContours, -1, (0, 255, 0), 2)
        
        
        #Of the biggest contours, sort them by closest to the center
        contourDistance = []
        for contour in topContours:
            # find center of each contour
            M = cv2.moments(contour)
            center_X = int(M["m10"] / M["m00"])
            center_Y = int(M["m01"] / M["m00"])
            contour_center = (center_X, center_Y)
        
            # calculate distance to direction center
            distances_to_center = (distance.euclidean(directionCenter, contour_center))
        
            # save to a list of dictionaries
            contourDistance.append({'contour': contour, 'center': contour_center, 'distance_to_center': distances_to_center})
        
        # sort the distances
        sortedDistances = sorted(contourDistance, key=lambda i: i['distance_to_center'])
    
        # find contour of closest building to center and draw it (green)
        try:
            centerContour = sortedDistances[0]['contour']
            cv2.drawContours(img, [centerContour], 0, (0, 255, 0), 2)
        except IndexError:
            print("contour error")
        
    else:
        cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    
    cv2.imshow('Video feed', img)
    
    if cv2.waitKey(1) == 13: # Exit on pressing the enter key
        break


cap.release()
cv2.destroyAllWindows()
