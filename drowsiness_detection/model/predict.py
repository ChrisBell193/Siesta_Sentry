from ultralytics import YOLO
#from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import os

def run_prediction():
    model = YOLO(
                os.path.join('data',
                            'detect',
                            'train',
                            'weights',
                            'best.pt'
                            )
    )

    results = model.predict(source='0', show=True, conf=0.7)
    print(results)
