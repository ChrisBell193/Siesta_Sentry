# 📚 Siesta Sentry

## Introduction
Have you ever been in a situation where you're driving at night, the darkness outside, and you start to feel that drowsiness creeping in? Your eyelids become heavy, and you find yourself nodding off, a potentially perilous situation when behind the wheel. Or perhaps, in today's world of online classes, you've struggled to stay awake during a lecture, fearing you might miss crucial information. What if there was a product that could automatically detect these moments of drowsiness or danger and provide a wake-up call? That's precisely what Siesta Sentry aims to achieve.

![Screenshot 2023-09-05 at 11 49 26](https://github.com/ChrisBell193/Siesta_Sentry/assets/138370119/60a12657-9987-4cda-94ae-8282a87e221c)
<br>

Check out the app for yourself:  https://siestasentry.streamlit.app/


## Description
#### Dataset
Utilized the Drowsiness Dataset from the University of Texas at Arlington, consisting of videos where volunteers recorded themselves in alert and drowsy states.
#### Data Preprocessing
Extracted frames from videos using OpenCV, ensuring uniform sizing and handling face proximity to the camera.
Augmented the dataset through techniques such as zooming and tilting.
#### Face Detection
Leveraged Mediapipe to locate faces in each frame, converting bounding box coordinates to x-center and y-center for model compatibility.
#### Model Training
Developed a YOLOv8 model using frame images, corresponding class labels, and bounding box values.
Tuned hyperparameters and upgraded to a larger YOLO version, achieving high recall and accuracy.
#### Deployment
Deployed the trained model to Streamlit Cloud, overcoming video capture challenges on the platform.
Siesta Sentry can now effectively detect drowsy faces in a live video feed.


## Getting Started
### Setup
Follow the steps below to run everything locally!

First, clone the repository and then create a new directory called data within the drowsiness_detection directory. Within that folder, create a directory called raw data, and within that directory another called video. Structure should look like this:

____________________________
<pre>
drowsiness_detection/
├── data/
│   └── raw_data/
│       └── video/
└── config/
│  
...
</pre>
Second, download the videos from this [link](https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset) and move all of the folders containing the videos from the different folds into the video directory you just created, such that your video directory has folders numbered from 1 to 48. each containing 3 video files. 

Third, install the requirements:
```
pip install -r requirements.txt
```

Next, you will need to go to the home/<user>/.config/Ultralytics directory and open settings.yaml and change the following three lines to read as follows, replacing <path/to> with the path to your Siesta_Sentry directory:
```
datasets_dir: /home/path/to/Siesta_Sentry
weights_dir: //home/path/to/Siesta_Sentry/data
runs_dir: /home/path/to/Siesta_Sentry/data
```

Lastly, in your commandline, run:
```
python main.py
```

Note: locally you will need a camera on source[0] that is accessible by OpenCV.  

## Acknowledgements
University of Texas at Arlington [Drowsiness Dataset](https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset)

## Team Members
- [Chris Bell](https://www.linkedin.com/in/chris-bell-1263171b3/)
- [Melody Nguyen](https://www.linkedin.com/in/melody-duong/)
- [Giovanni Jolard](https://www.linkedin.com/in/giovanni-jolard-3b9b721b7/)
- [Joshua Higgins](https://www.linkedin.com/in/joshua-higgins-29ab4028b/)
