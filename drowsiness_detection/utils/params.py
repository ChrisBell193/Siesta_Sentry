import os
import numpy as np

##################  CONSTANTS  #####################
PARENT_DIRECTORY = 'data'
DATA_DIRECTORY = os.path.join(PARENT_DIRECTORY,"raw_data")
VIDEO_DIRECTORY = os.path.join(DATA_DIRECTORY, 'video')

CLASSES_NUMS = ['0','5','10']
CLASSES_DICT = {
    '0': 'alert',
    '5': 'normal',
    '10': 'drowsy'
}

SPECIAL_GUYS = [23,42]
TRAIN_SET_RANGE = range(1,43)
TRAIN_SET_NORMAL = [num for num in TRAIN_SET_RANGE if num not in SPECIAL_GUYS]
VAL_SET_RANGE = range(43,49)
NUMBER_OF_FRAMES =20

MODEL_TYPE = "yolov8n.yaml"
CONFIG_FILE = "drowsiness_detection/config/google_colab_config_augment_1.yaml"
EPOCHS = 100
PATIENCE = 50
BATCH_SIZE = 16
