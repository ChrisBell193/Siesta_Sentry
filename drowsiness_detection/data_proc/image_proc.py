import uuid   # Unique identifier
import os
import time
import cv2
from PIL import Image
import math
from typing import Tuple, Union
import numpy as np
from drowsiness_detection.utils.params import *


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


def shrink_and_add_padding(image_path, output_path, target_percentage = 0.8):
    """
    Shrink an image to target_percentage of its size
    and add black padding to keep the original size.

    Args:
        image_path (str): Path to the input image file.
        output_path (str): Path to save the shrunken and padded image.

    Returns:
        None
    """
    # Open the image
    image = Image.open(image_path)

    # Get the original width and height of the image
    width, height = image.size

    # Calculate the new dimensions after shrinking to 80%
    new_width = int(width * target_percentage)
    new_height = int(height * target_percentage)

    # Resize the image to 80% of its original size
    shrunk_image = image.resize((new_width, new_height), Image.BILINEAR)

    # Calculate the padding dimensions
    padding_x = (width - new_width) // 2
    padding_y = (height - new_height) // 2

    # Create a new image with black background
    padded_image = Image.new("RGB", (width, height), (0, 0, 0))

    # Paste the shrunken image onto the padded image at the center
    padded_image.paste(shrunk_image, (padding_x, padding_y))

    # Save the padded image
    padded_image.save(output_path)

def collect_images_with_webcam(output_directory, classes, number_imgs, person_ids, delay_time=3):
    cap = cv2.VideoCapture(0)

    # Loop through labels
    for label in classes:
        print('Collecting images for {}'.format(label))
        time.sleep(5)

        # Loop through image range
        for img_num in range(number_imgs):
            print('Collecting images for {}, image number {}'.format(label, img_num))

            # Webcam feed
            ret, frame = cap.read()

            if not ret:
                print("Error reading webcam frame")
                continue  # Skip this iteration and proceed to the next loop iteration

            # Naming out image path
            imgname = os.path.join(output_directory, label+'_'+str(person_ids)+'_'+str(uuid.uuid1())+'.jpg')

            # Writes out image to file
            cv2.imwrite(imgname, frame)

            # Render to the screen
            cv2.imshow('Image Collection', frame)

            # delay between captures
            time.sleep(delay_time)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
