import cv2 # needed to call optical flow algorithm
import numpy as np # needed for optical flow algorithm

from tkinter import filedialog as fd # needed to grab video file
import os # needed for file name manipulation
import re # needed for regular expression split
import datetime # needed for timestamps

import csv # needed to write to csv file

def calculate_camera_motion(previous_frame, current_frame):
    """
    Calculates camera motion using optical flow.

    Args:
        previous_frame: The previous frame (grayscale).
        current_frame: The current frame (grayscale).

    Returns:
        A tuple containing:
            - Translation vector (tx, ty).
            - Rotation angle (in degrees).
    """

    # Feature detection using goodFeaturesToTrack
    points1 = cv2.goodFeaturesToTrack(previous_frame, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

    # Calculate optical flow using Lucas-Kanade method
    points2, status, err = cv2.calcOpticalFlowPyrLK(previous_frame, current_frame, points1, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Select good points
    good_points1 = points1[status == 1]
    good_points2 = points2[status == 1]

   # Estimate the homography matrix using RANSAC
    if len(good_points1) > 4:
        H, mask = cv2.findHomography(good_points1, good_points2, cv2.RANSAC, 5.0)
    else:
        return (0, 0), 0  # Not enough points to estimate motion

    # Decompose the homography matrix to get rotation and translation
    if H is not None:
      
        # Extract translation
        tx = H[0, 2]
        ty = H[1, 2]

        # Extract rotation (simplified for 2D)
        rotation_angle = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi

        return (tx, ty), rotation_angle
    else:
         return (0, 0), 0

# Pick a video to test
#video_path = "./videos/lv-flyover.mp4"
#video_path = "./videos/utah-flyover.mp4"
#video_path = "./videos/wa-flyover.mp4"

# pull video path from file dialog
video_path = fd.askopenfilename(title="Select a video to process")

#print(video_path)
fileName = os.path.basename(video_path).split('.')[0]
#print(fileName)
nameBreak = re.split(r'[-_,;]', fileName)

startTime = datetime.datetime(int(nameBreak[0]), int(nameBreak[1]), int(nameBreak[2]), int(nameBreak[3]), int(nameBreak[4]), int(nameBreak[5]), int(nameBreak[6]))

#print(startTime)

cap = cv2.VideoCapture(video_path)

# get the time between frames
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1 / fps
counter = 0

ret, previous_frame = cap.read()
previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

with open(f"{video_path}_camera_motion.csv", mode='w', newline='') as file:
    fieldNames = ['Time', 'X', 'Y', 'Rotation']
    writer = csv.DictWriter(file, fieldnames=fieldNames)
    writer.writeheader()

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        newTime = startTime + datetime.timedelta(seconds=counter * frame_time)

        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        (tx, ty), rotation_angle = calculate_camera_motion(previous_frame, current_frame_gray)

        #print(f"Translation: x={tx:.2f}, y={ty:.2f}, Rotation: {rotation_angle:.2f} degrees")
        frameText = f"Camera Translation: " + "\n" + f"X= {tx:.2f}" + "\n"  + f"Y= {ty:.2f}" + "\n" + f"Rotation: {rotation_angle:.2f} degrees"

        # write to csv file
        #print(f"{str(newTime)} {tx} {ty} {rotation_angle}")
        writer.writerow({'Time': newTime, 'X': tx, 'Y': ty, 'Rotation': rotation_angle})

        # Split the text into lines
        lines = frameText.split('\n')

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Initial position
        x, y = 50, 50

        # Line height (adjust as needed)
        line_height = 30

        # Render each line
        for line in lines:
            cv2.putText(current_frame, line, (x, y), font, 1, (0, 255, 255), 2)
            y += line_height  # Move to the next line position

        # Use putText() method for inserting text on video 
        #cv2.putText(current_frame, frameText, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4) 

        previous_frame = current_frame_gray

        # Display the resulting frame (optional)
        cv2.imshow('Camera Motion Tracking', current_frame)

        # increment the counter
        counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()