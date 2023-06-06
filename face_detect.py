import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

model_path = 'model/blaze_face_short_range.tflite'

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Callback function to print face detection results
def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    print('face detector result: {}'.format(result))

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)


# Loop through each frame in the video
frame_index = 0
with FaceDetector.create_from_options(options) as detector:
    # OpenCV initialization
    video_capture = cv2.VideoCapture(0)

    # Get the frame rate of the video
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    while True:
        # Read the latest frame from the webcam
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to capture frame from webcam.")
            break

        # Convert the frame received from OpenCV to a MediaPipe's Image object.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            width=frame.shape[1],
            height=frame.shape[0],
            data=frame.tobytes(),
        )

        # Calculate the frame timestamp in milliseconds
        frame_timestamp_ms = int(frame_index * (1000 / fps))
        frame_index += 1

        # Send live image data to perform face detection.
        # The results are accessible via the `result_callback` provided in
        # the `FaceDetectorOptions` object.
        # The face detector must be created with the live stream mode.
        detector.detect_async(mp_image, frame_timestamp_ms)

        # Display the frame
        cv2.imshow("Webcam", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()
