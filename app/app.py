import streamlit as st
import torch
import numpy as np
import cv2
import time
from pytube import YouTube
from PIL import Image
import os
import pygame
from ultralytics import YOLO
# from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import sys
from drowsiness_detection.utils.params import *

# Load the YOLOv5 model
if 'model' not in st.session_state:
    st.session_state['model'] = YOLO(
        os.path.join(PARENT_DIRECTORY,
                     'runs',
                     'BEST_runs_10th_tuned_with_pics_of_8',
                     'detect',
                     'train5',
                     'weights',
                     'best.pt'
                     )
    )
model = st.session_state['model']

# Install ffprobe alert_sound.wav
# 211C3E
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model = YOLO('/home/giovanni/code/GiovanniJolard/test_yolo_streamlit/runs_6/detect/yolov8s/weights/best.pt')
# Initialization
# model = YOLO('/home/giovanni/code/GiovanniJolard/test_yolo_streamlit/runs_6/detect/yolov8s/weights/best.pt')
# Streamlit page configuration
st.set_page_config(
    page_title="Siesta Sentry",
    page_icon=":male-detective:",
    # layout="wide",
    initial_sidebar_state="expanded"
)

# Load the image for the title
title_image = Image.open(os.path.join(
    PARENT_DIRECTORY,
    'logo',
    'SIESTA8LOGO.png'
))

# Display the title image
st.image(title_image, use_column_width=True)

# Load the alert sound
alert_sound_path = os.path.join(PARENT_DIRECTORY,
                                'alarm_sound',
                                'alarm.wav')

# Initialize pygame mixer
pygame.mixer.init()

# Input field for YouTube video URL or video file path
st.write("\n\n")
video_input_option = st.radio("Choose an option:", ("YouTube Video URL", "Video File Path"))
if video_input_option == "YouTube Video URL":
    video_url = st.text_input("Enter the YouTube video URL:")
    video_path = False
else:
    video_url = False
    video_path = st.file_uploader("Upload a video", type=["mp4","mov"])

# Check if the user submitted a video URL or uploaded a video
if st.button("Start Detection") and (video_url or video_path):
    try:
        if video_url:
            # Download the YouTube video
            yt = YouTube(video_url)
            video_stream = yt.streams.filter(file_extension="mp4").first()
            video_path = video_stream.download()
        elif video_path is not None:
            # Temporarily save the uploaded file
            with open(video_path.name, "wb") as f:
                f.write(video_path.read())
            video_path = video_path.name

        # Open the video for streaming
        cap = cv2.VideoCapture(str(video_path))
        video_elem = st.empty()
        running = True
        stop_button = st.button("Stop Detection")
        drowsy_counter = 0
        threshold_frames = 5
        last_alert_time = 0

        while running:
            ret, frame = cap.read()
            if not ret or stop_button:
                running = False
                break

            # # Perform detection
            results = model(frame)
            for result in results:
                names_dict = result.names
                if 'drowsy' in names_dict.values():
                    drowsy_counter += 1
                    if drowsy_counter >= threshold_frames:
                        current_time = time.time()
                        if current_time - last_alert_time >= 20:
                            # Play the alert sound
                            pygame.mixer.Sound(alert_sound_path).play()
                            last_alert_time = current_time
                            st.info("Drowsiness Detected!")
                else:
                    drowsy_counter = 0

                # Use the result.orig_img attribute to display the original image
                detection_image = np.array(result.orig_img)

                # Draw detection boxes on the image (if result.boxes contains coordinates)
                if result.boxes is not None:
                    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                        x1, y1, x2, y2 = box.tolist()
                        class_name = names_dict[int(cls)]
                        if class_name == "alert":
                            label = f"{class_name} ({conf:.2f})"
                            # Draw detection box and label
                            cv2.rectangle(detection_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(detection_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        elif class_name == "drowsy":
                            label = f"{class_name} ({conf:.2f})"
                            # Draw detection box and label
                            cv2.rectangle(detection_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255, 2))
                            cv2.putText(detection_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Display the detection image
                video_elem.image(detection_image, channels="BGR", use_column_width=True)
                cv2.waitKey(1)
                # st_video_with_fixed_width(detection_image, target_width)
                # cv2.waitKey(1)

        # Display a message when detection is stopped
        st.write("Detection stopped")

    except Exception as e:
        # Display an error message if there's a problem
        st.error("An error occurred while processing the video.")
        st.error(str(e))
