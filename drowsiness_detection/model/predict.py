from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import os


PARENT_DIRECTORY = os.path.join(os.path.expanduser('~'), "code", "ChrisBell193", "Siesta_Sentry")
model = YOLO(
            os.path.join(PARENT_DIRECTORY,
                         'all_runs',
                        'BEST_runs_10th_tuned_with_pics_of_8',
                        'detect',
                        'train5',
                        'weights',
                        'best.pt'
                        )
)

results = model.predict(source='0', show=True, conf=0.3)
print(results)
