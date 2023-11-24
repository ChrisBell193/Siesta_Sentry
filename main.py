import uuid   # Unique identifier
import os
import time
# import cv2
# from PIL import Image
import math
from typing import Tuple, Union
# import numpy as np
import shutil
from drowsiness_detection.data_proc.image_proc import *
from drowsiness_detection.data_proc.image_annot import *
from drowsiness_detection.utils.params import *
from drowsiness_detection.data_proc.video_to_img import video_to_frames
from drowsiness_detection.model.train import train
from drowsiness_detection.model.predict import run_prediction


def image_preprocess():

    #STEP 1: Preprocessing for train set normal, excep person_id 23 and 42
    print('Preprocessing for training set, except person_id 23 and 42')
    print(f'Getting {NUMBER_OF_FRAMES} for each video')
    video_to_frames(video_range=TRAIN_SET_NORMAL,
                    folder= 'train',
                    num_frames=NUMBER_OF_FRAMES)

    print(f'Image processing...')
    train_images_path = os.path.join(DATA_DIRECTORY,
                                     f'raw_image_{NUMBER_OF_FRAMES}',
                                     'train'
                                     )
    train_images_processed_path = os.path.join(DATA_DIRECTORY,
                                               'train',
                                               'images')
    if not os.path.exists(train_images_processed_path):
        os.makedirs(train_images_processed_path)

    for root, dirs, files in os.walk(train_images_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            resize_and_fill(input_path = file_path,
                    output_path= os.path.join(train_images_processed_path, file_name),
                    size=640,
                    fill_color=(0, 0, 0))
    print(f'Finish processing for normal train set')

    #STEP 2: Preprocessing for person_id 23
    print('Preprocessing for person_id 23')
    print(f'Getting {NUMBER_OF_FRAMES} for each video')
    video_to_frames(video_range=[SPECIAL_GUYS[0]],
                    folder= str(SPECIAL_GUYS[0]),
                    num_frames=NUMBER_OF_FRAMES)

    print(f'Image processing...')
    train_23_path = os.path.join(DATA_DIRECTORY,
                                     f'raw_image_{NUMBER_OF_FRAMES}',
                                     str(SPECIAL_GUYS[0]))
    for root, dirs, files in os.walk(train_23_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            crop_to_square(image_path= file_path,
                   output_path= os.path.join(train_images_processed_path, file_name),
                   target_size=640)
    print(f'Finish processing for person_id {SPECIAL_GUYS[0]}')

    #STEP 3: Preprocessing for person_id 42
    print('Preprocessing for person_id 42')
    print(f'Getting {NUMBER_OF_FRAMES} for each video')
    video_to_frames(video_range=[SPECIAL_GUYS[1]],
                    folder= str(SPECIAL_GUYS[1]),
                    num_frames=NUMBER_OF_FRAMES)

    print(f'Image processing...')
    train_42_path = os.path.join(DATA_DIRECTORY,
                                     f'raw_image_{NUMBER_OF_FRAMES}',
                                     str(SPECIAL_GUYS[1]))
    for root, dirs, files in os.walk(train_42_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            shrink_and_add_padding(image_path= file_path,
                           output_path= os.path.join(train_images_processed_path, file_name),
                           target_percentage = 0.8)
    print(f'Finish processing for person_id {SPECIAL_GUYS[1]}')

    #STEP4: Preprocessing for val set
    print('Preprocessing for validation set')
    print(f'Getting {NUMBER_OF_FRAMES} for each video')
    video_to_frames(video_range=VAL_SET_RANGE,
                    folder= 'val',
                    num_frames=NUMBER_OF_FRAMES)

    print(f'Image processing...')
    val_images_path = os.path.join(DATA_DIRECTORY,
                                     f'raw_image_{NUMBER_OF_FRAMES}',
                                     'val')
    val_images_processed_path = os.path.join(DATA_DIRECTORY,
                                               'val',
                                               'images')
    if not os.path.exists(val_images_processed_path):
        os.makedirs(val_images_processed_path)

    for root, dirs, files in os.walk(val_images_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            resize_and_fill(input_path = file_path,
                    output_path= os.path.join(val_images_processed_path, file_name),
                    size=640,
                    fill_color=(0, 0, 0))
    print('Finish processing for validation set')
    print('Finish image preprocessing')

def image_annotate():
    train_images_path = os.path.join(DATA_DIRECTORY,
                                'train',
                                'images'
                                )
    train_labels_path = os.path.join(DATA_DIRECTORY,
                                'train',
                                'labels'
                                )

    val_images_path = os.path.join(DATA_DIRECTORY,
                                'val',
                                'images'
                                )
    val_labels_path = os.path.join(DATA_DIRECTORY,
                                'val',
                                'labels'
                                )

    #STEP 1: Annotate train folder
    if not os.path.exists(train_labels_path):
        os.makedirs(train_labels_path)

    print('Saving label files for train set')
    annotate_and_save_labels_file(input_directory= train_images_path,
                                  output_labels_directory= train_labels_path)

    #STEP 2: Annotate validation folder
    if not os.path.exists(val_labels_path):
        os.makedirs(val_labels_path)

    print('Saving label files for val set')
    annotate_and_save_labels_file(input_directory= val_images_path,
                                  output_labels_directory= val_labels_path)

    print('Finish annotating pictures, label files are saved')


#Calling all functions in order

#Screencapture videos and save them
image_preprocess()

#Generate .txt files to use during training
image_annotate()

#Train the YOLO model
train(model_type=MODEL_TYPE, config_file=CONFIG_FILE, epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE)

#Open local camera and show predictions live
run_prediction()
