from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import os

model = YOLO(
            os.path.join('all_runs',
                'BEST_runs_10th_tuned_with_pics_of_8',
                'detect',
                'train5',
                'weights',
                'best.pt'
                )
)

results = model.predict(source='0', show=True, conf=0.7)
print(results)
