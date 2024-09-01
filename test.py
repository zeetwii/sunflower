from ultralytics import YOLO # needed for YOLO model
import cv2 # needed for computer vision processes
import math # needed for math functions
import time # needed for sleep

# Simple test tracking code

# Pick a model to test
#model = YOLO("./models/yolov8s-worldv2.pt")
model = YOLO("./models/yolov10n.pt")


# Pick a video to test
video_path = "./videos/lv-flyover.mp4"
#video_path = "./videos/utah-flyover.mp4"
#video_path = "./videos/wa-flyover.mp4"

cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.predict(frame, conf=0.1)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()