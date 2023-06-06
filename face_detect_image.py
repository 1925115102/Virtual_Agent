import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import visualization

IMAGE_FILE = 'test_1.png'

# Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='model/blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# Load the input image.
image = mp.Image.create_from_file(IMAGE_FILE)

# Detect faces in the input image.
detection_result = detector.detect(image)

# Process the detection result. In this case, visualize it.
image_copy = np.copy(image.numpy_view())
annotated_image = visualization.visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
cv2.imshow('Image',rgb_annotated_image)

output_file = 'annotated_image.png'
cv2.imwrite(output_file, rgb_annotated_image)

print(f"Annotated image saved as '{output_file}'")
