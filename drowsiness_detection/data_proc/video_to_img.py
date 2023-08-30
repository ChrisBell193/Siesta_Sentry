import uuid   # Unique identifier
import os
import cv2
from typing import Tuple, Union
import numpy as np
from drowsiness_detection.utils.params import *

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


def mp4_to_frames(video_range: Tuple, train_or_val: str, num_frames = 50):
    """
    get pictures from mp4
    """
    for person_id in video_range:
        video_folder = os.path.join(VIDEO_DIRECTORY, str(person_id))
        for class_num in CLASSES_NUMS:
            video_path = os.path.join(video_folder, class_num + '.mp4' )
            image_path = os.path.join(DATA_DIRECTORY, f'raw_image_{num_frames}', train_or_val)
            if not os.path.exist(image_path):
                os.makedir(image_path)
            extract_equally_spaced_frames(video_path,
                                          image_path,
                                          num_frames=num_frames,
                                          status_id=CLASSES_DICT[class_num],
                                          person_id=person_id)

def mov_to_frames(video_range: list, train_or_val: str, num_frames = 50):
    """
    get pictures from mp4
    """
    for person_id in video_range:
        video_folder = os.path.join(VIDEO_DIRECTORY, str(person_id))
        for class_num in CLASSES_NUMS:
            video_path = os.path.join(video_folder, class_num + '.mov' )
            image_path = os.path.join(DATA_DIRECTORY, f'raw_image_{num_frames}', train_or_val)
            if not os.path.exist(image_path):
                os.makedir(image_path)
            extract_equally_spaced_frames(video_path,
                                          image_path,
                                          num_frames=num_frames,
                                          status_id=CLASSES_DICT[class_num],
                                          person_id=person_id)

def m4v_to_frames(video_range: list, train_or_val: str, num_frames = 50):
    """
    get pictures from mp4
    """
    for person_id in video_range:
        video_folder = os.path.join(VIDEO_DIRECTORY, str(person_id))
        for class_num in CLASSES_NUMS:
            video_path = os.path.join(video_folder, class_num + '.m4v' )
            image_path = os.path.join(DATA_DIRECTORY,
                                      f'raw_image_{num_frames}',
                                      train_or_val)
            if not os.path.exist(image_path):
                os.makedir(image_path)
            extract_equally_spaced_frames(video_path,
                                          image_path,
                                          num_frames=num_frames,
                                          status_id=CLASSES_DICT[class_num],
                                          person_id=person_id)

def video_to_frames(video_range: list, folder: str, num_frames = 20):
    mp4_to_frames(video_range, num_frames,folder)
    mov_to_frames(video_range, num_frames,folder)
    m4v_to_frames(video_range, num_frames,folder)
    print(f'Finish getting {num_frames} of video for {folder} set')
