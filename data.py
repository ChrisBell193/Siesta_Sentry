import uuid   # Unique identifier
import os
import time
import cv2
from PIL import Image


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
