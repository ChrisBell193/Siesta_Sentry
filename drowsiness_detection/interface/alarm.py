import cv2
import os
import numpy as np
from pygame import mixer
import time
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import os
from drowsiness_detection.utils.params import *

mixer.init()
sound = mixer.Sound('alarm.wav')

# PARENT_DIRECTORY = os.path.join(os.path.expanduser('~'), "code", "ChrisBell193", "Siesta_Sentry")
model = YOLO(
            os.path.join(PARENT_DIRECTORY,
                         'runs',
                        'BEST_runs_10th_tuned_with_pics_of_8',
                        'detect',
                        'train5',
                        'weights',
                        'best.pt'
                        )
)

results = model.predict(source='0', show=True, conf=0.7)
