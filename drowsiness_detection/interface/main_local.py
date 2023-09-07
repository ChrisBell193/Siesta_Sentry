import uuid   # Unique identifier
import os
import time
import cv2
from PIL import Image
import math
from typing import Tuple, Union
import numpy as np
import shutil
from drowsiness_detection.data_proc.image_proc import *
from drowsiness_detection.data_proc.image_annot import *
from drowsiness_detection.utils.params import *
from drowsiness_detection.data_proc.video_to_img import video_to_frames



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
                                               f'processed_images_{NUMBER_OF_FRAMES}',
                                               'train',
                                               'images')
    if not os.path.exist(train_images_processed_path):
        os.makedir(train_images_processed_path)

    resize_and_fill(input_path = train_images_path,
                    output_path= train_images_processed_path,
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
                                     SPECIAL_GUYS[0])
    train_23_processed_path = os.path.join(DATA_DIRECTORY,
                                               f'processed_images_{NUMBER_OF_FRAMES}',
                                               SPECIAL_GUYS[0])
    if not os.path.exist(train_23_processed_path):
        os.makedir(train_23_processed_path)
    crop_to_square(image_path= train_23_path,
                   output_path= train_23_processed_path,
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
                                     SPECIAL_GUYS[1])
    train_42_processed_path = os.path.join(DATA_DIRECTORY,
                                               f'processed_images_{NUMBER_OF_FRAMES}',
                                               SPECIAL_GUYS[1])
    if not os.path.exist(train_42_processed_path):
        os.makedir(train_42_processed_path)
    shrink_and_add_padding(image_path= train_42_path,
                           output_path= train_42_processed_path,
                           target_percentage = 0.8)
    print(f'Finish processing for person_id {SPECIAL_GUYS[1]}')

    #STEP 4: move processed images of 23 and 42 to train folder
    print('Moving processed images of 23 and 42 to train folder')
    # Move files from train_23_processed_path to train_images_processed_path
    for filename in os.listdir(train_23_processed_path):
        src_23 = os.path.join(train_23_processed_path, filename)
        dst_23 = os.path.join(train_images_processed_path, filename)
        shutil.move(src_23, dst_23)

    # Move files from train_42_processed_path to train_images_processed_path
    for filename in os.listdir(train_42_processed_path):
        src_42 = os.path.join(train_42_processed_path, filename)
        dst_42 = os.path.join(train_images_processed_path, filename)
        shutil.move(src_42, dst_42)

    # Remove train_23_processed_path and train_42_processed_path directories
    os.rmdir(train_23_processed_path)
    os.rmdir(train_42_processed_path)
    print('Finish processing images for train dataset')

    #STEP5: Preprocessing for val set
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
                                               f'processed_images_{NUMBER_OF_FRAMES}',
                                               'val',
                                               'images')
    if not os.path.exist(val_images_processed_path):
        os.makedir(val_images_processed_path)

    resize_and_fill(input_path = val_images_path,
                    output_path= val_images_processed_path,
                    size=640,
                    fill_color=(0, 0, 0))
    print('Finish processing for validation set')
    print('Finish image preprocessing')

def image_annotate():
    train_images_path = os.path.join(DATA_DIRECTORY,
                                f'processed_images_{NUMBER_OF_FRAMES}',
                                'train',
                                'images'
                                )
    train_labels_path = os.path.join(DATA_DIRECTORY,
                                f'processed_images_{NUMBER_OF_FRAMES}',
                                'train',
                                'labels'
                                )

    val_images_path = os.path.join(DATA_DIRECTORY,
                                f'processed_images_{NUMBER_OF_FRAMES}',
                                'val',
                                'images'
                                )
    val_labels_path = os.path.join(DATA_DIRECTORY,
                                f'processed_images_{NUMBER_OF_FRAMES}',
                                'val',
                                'labels'
                                )

    #STEP 1: Annotate train folder
    print('Saving label files for train set')
    annotate_and_save_labels_file(input_directory= train_images_path,
                                  output_labels_directory= train_labels_path)

    #STEP 1: Annotate validation folder
    print('Saving label files for val set')
    annotate_and_save_labels_file(input_directory= val_images_path,
                                  output_labels_directory= val_labels_path)

    print('Finish annotating pictures, label files are saved')
