import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import visualization

# Load the video using OpenCV's VideoCapture
video_path = 'video_test_3.mov'
cap = cv2.VideoCapture(video_path)

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a FaceDetector object
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='model/blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.VIDEO)
detector = vision.FaceDetector.create_from_options(options)




# Loop through each frame in the video
frame_index = 0
while cap.isOpened():
    # Read the frame from the video
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if no frame is retrieved

    # Convert the frame to MediaPipe's Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Calculate the frame timestamp in milliseconds
    frame_timestamp_ms = int(frame_index * (1000 / fps))
    frame_index += 1

    # Perform face detection on the provided single image
    face_detector_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    # Draw bounding boxes and keypoints on the frame
    annotated_image = visualization.visualize(frame, face_detector_result)

    # Display the frame (optional)
    cv2.imshow('Frame', annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Break the loop if 'q' key is pressed

# Release the VideoCapture and close any open windows
cap.release()
cv2.destroyAllWindows()
