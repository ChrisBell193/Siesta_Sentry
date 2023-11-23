from ultralytics import YOLO
#from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import os

def run_prediction():
    #TODO update this
    PARENT_DIRECTORY = os.path.join(os.path.expanduser('~'), "code", "ChrisBell193", "Siesta_Sentry")
    model = YOLO(
    #TODO update this
                os.path.join(PARENT_DIRECTORY,
                            'runs',
                            'should be best fine tuned',
                            'weights',
                            'best.pt'
                            )
    )

    results = model.predict(source='0', show=True, conf=0.7)
    print(results)
