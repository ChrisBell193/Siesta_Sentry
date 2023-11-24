import uuid   # Unique identifier
import os
import time
import cv2
from PIL import Image
import math
from typing import Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                        math.isclose(1, value))

    # Check if both normalized_x and normalized_y are within valid range (0 to 1)
    # If one or both values are outside the valid range, return None
    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
        return None

    # Convert normalized values to pixel coordinates
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def visualize(
    image,
    detection_result,
    MARGIN = 10,  # pixels
    ROW_SIZE = 10 , # pixels
    FONT_SIZE = 1,
    FONT_THICKNESS = 1,
    TEXT_COLOR = (255, 0, 0)  # red
) -> np.ndarray:
    """
    Draws bounding boxes and keypoints on the input image and return it.
    Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
    Returns:
    Image with bounding boxes.
    """
    # Create a copy of the input image to draw annotations on
    annotated_image = image.copy()
    # Get the height, width, and number of color channels of the image
    height, width, _ = image.shape

    # Loop through each detection in the result
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        # Draw a rectangle around the detected object
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints associated with the detection
        for keypoint in detection.keypoints:
            # Convert normalized keypoint coordinates to pixel coordinates
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                            width, height)
            color, thickness, radius = (0, 255, 0), 2, 2
            # Draw a circle at the keypoint location
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Get the category information for the detection
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                        MARGIN + ROW_SIZE + bbox.origin_y)
    # Draw the category label and probability score
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image

def annotate_and_save_labels_file(input_directory, output_labels_directory):
    """
    Annotate picture and save the labels file in txt with same name with image
    """
    # Iterate through all files in the input directory
    for file_names in os.listdir(input_directory):
        if file_names.endswith('.jpg'):
            # Get the full path to the input image
            path_image = os.path.join(input_directory, file_names)
            # Initialize face detection
            mp_face_detect = mp.solutions.face_detection
            face_detect = mp_face_detect.FaceDetection(min_detection_confidence=.6)
            img_guy = cv2.imread(path_image)
            results_lists = face_detect.process(img_guy)

            # Process the detected faces
            if results_lists.detections is not None:
                for result_list in results_lists.detections:
                    bbx = result_list.location_data.relative_bounding_box
                    x_center = bbx.xmin + (bbx.width / 2)
                    y_center = bbx.ymin + (bbx.height / 2)
                    width = bbx.width
                    height = bbx.height

                    # Prepare the path for saving the label file
                    save_path = os.path.join(output_labels_directory)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    file_object = open((os.path.join(save_path, f"{file_names.removesuffix('.jpg')}.txt")), "w")

                    # Write label information based on file name
                    if file_names.startswith('a'):
                        file_object.write(f"0 {x_center} {y_center} {width} {height}")
                    elif file_names.startswith('n'):
                        file_object.write(f"1 {x_center} {y_center} {width} {height}")
                    elif file_names.startswith('d'):
                        file_object.write(f"2 {x_center} {y_center} {width} {height}")
                    else:
                        print("this is a mistake")
                    file_object.close()
            else:
                print('no faces detected')

def face_dectection_bbx_for_picture(input_folder, output_folder):
    """
    Adding bounding box for face detection and save picture to folder
    """
    base_options = python.BaseOptions(model_asset_path='detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    for filenames in os.listdir(input_folder):
        for filename in filenames:
            if filename.endswith('.jpg'):
                image_path = os.path.join(input_folder, filename)
                # Load the input image.
                image = mp.Image.create_from_file(image_path)

                # Detect faces in the input image.
                detection_result = detector.detect(image)

                # Process the detection result. Visualize it, save it.
                image_copy = np.copy(image.numpy_view())
                annotated_image = visualize(image_copy, detection_result)
                rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                reversed_rgb = cv2.cvtColor(rgb_annotated_image, cv2.COLOR_BGR2RGB)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                plt.imsave((os.path.join(output_folder, filename)), reversed_rgb)
