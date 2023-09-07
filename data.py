import uuid   # Unique identifier
import os
import time
import cv2
from PIL import Image
import math
from typing import Tuple, Union
import numpy as np


def extract_equally_spaced_frames(video_path, output_folder, status_id, person_id, num_frames=21):
    """
    Define a function to extract equally spaced frames
    from a video file and save them as images
    """
    # Open the video file using OpenCV's VideoCapture
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the frame rate (frames per second) of the video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the number of frames to skip the first 10 seconds
    skip_frames = int(frame_rate * 10)

    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop to extract and save the specified number of equally spaced frames
    for i in range(num_frames):
        # Calculate the index of the frame to be extracted
        frame_idx = skip_frames + (frame_count - skip_frames) * i // num_frames

        # Set the VideoCapture object's position to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read the frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if ret:
            # Generate a unique filename based on status ID, person ID, and a UUID
            output_path = (os.path.join(output_folder,
                                        status_id
                                        + '_'
                                        + person_id
                                        + '_'
                                        + str(uuid.uuid1())
                                        + '.jpg')
            )

            # Save the frame as an image
            cv2.imwrite(output_path, frame)

        # Check if the 'q' key is pressed to break the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object to free up resources
    cap.release()

def resize_and_fill(input_path, output_path, size=640, fill_color=(0, 0, 0)):
    """
    Add padding spaces to make pitures square shape
    Resize pictures to desized size
    """
    # Open the image
    image = Image.open(input_path)
    # Calculate the aspect ratio of the image
    aspect_ratio = image.width / image.height
    # Determine the new dimensions while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = size
        new_height = int(size / aspect_ratio)
    else:
        new_width = int(size * aspect_ratio)
        new_height = size
    # Resize the image while maintaining the aspect ratio
    resized_image = image.resize((new_width, new_height), resample=Image.BILINEAR)
    # Create a new image with the specified background color
    new_image = Image.new("RGB", (size, size), fill_color)
    # Calculate the position to paste the resized image
    paste_position = ((size - new_width) // 2, (size - new_height) // 2)
    # Paste the resized image onto the new image
    new_image.paste(resized_image, paste_position)
    # Save the final image. Quality=100 because they are already compressed images from taking screenshots.
    new_image.save(output_path, format="JPEG", quality=100)


def crop_to_square(image_path,output_path, target_size=640):
    """
    Crop the center part of an image to a square and resize it to the target size.

    Args:
        image_path (str): Path to the input image file.
        output_path (str): Path to save the cropped and resized image.
        target_size (int, optional): The size (width and height) of the output square image.
                                     Default is 640 pixels.
    Save cropped image to output path
    """
    # Open the image
    image = Image.open(image_path)

    # Get the original width and height of the image
    width, height = image.size

    # Calculate the cropping coordinates to keep the center part
    new_width = new_height = min(width, height)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    # Crop the image to the calculated coordinates
    cropped_image = image.crop((left, top, right, bottom))
    # Resize the cropped image to the target size
    resized_image = cropped_image.resize((target_size, target_size), Image.BILINEAR)
    resized_image.save(output_path, format="JPEG", quality=100)

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
