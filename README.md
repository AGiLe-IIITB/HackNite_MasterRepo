## Track
Machine Learning

## Contributors
Arismita Mukherjee, Gautam Prasanna Kappagal, Talla Likitha Reddy

## Problem Statement
Tackling terrorist activities with minimal human intervention using machine learning technology.

## Goal
To simulate a real-life scenario where we eliminate terrorist threats and free our captive soldiers using machine learning technology.

## Other repos under this project
- [Morse Code Reader](https://github.com/AGiLe-IIITB/morse_code)
- [Object Detection](https://github.com/AGiLe-IIITB/object_detection)
- [Finger Tracker](https://github.com/AGiLe-IIITB/finger_tracker)
- [Soldier Image Website](https://github.com/AGiLe-IIITB/soldier_image_web)
- [Audio to text](https://github.com/AGiLe-IIITB/audio_to_text)
- [Face Detection](https://github.com/AGiLe-IIITB/face_detection)

## Features
Our project utilizes blink detection for Morse code translation, object detection to detect buildings in a landscape, finger tracking for a virtual gun, audio-to-text conversion, and NLP to translate speech and detect keywords and terrorist threats in speech and face detection to distinguish enemies and comrades.

## Tech Stack
Programming Language: Python

Libraries used:
1. Morse Code: `scipy`, `imutils`, `numpy`, `argparse`, `dlib`, `OpenCV (cv2)`, `pynput (keyboard)`, `shape_predictor_68_face_landmarks.dat`
2. Object Detection: `yaml`, `argparse`, `csv`, `os`, `platform`, `sys`, `pathlib`, `torch`, `math`, `random`, `subprocess`, `datetime`
3. Finger Tracker: `mediapipe`, `OpenCV (cv2)`, `numpy`, `pyautogui`
4. Audio To Text/NLP: `speech_recgonition`, `re`, `dateutil`, `yake`
5. Terrorist Shooting Simulation Webpage: `HTML`, `CSS`, `JavaScript`
6. Face Detection: `sklearn (scikit learn)`, `imutils`, `numpy`, `argparse`, `pickle`, `OpenCV (cv2)`, `sys`, `dlib`, `time`, `face_recognition`, `mmod_human_face_detector.dat`

## How To Run
1. Morse Code:

We have a main Python code blink_morse1.py and two sub-codes `constants.py` and `morse_code.py` which are linked to the main code. We also have to link the `shape_predictor_68_face_landmarks` file and a pre-recorded video.

Terminal command: 
```
python blink_morse1.py -p shape_predictor_68_face_landmarks.dat -v face_gautam5_morse.mp4
```

2. Building Detection:

We have two main codes. The first code - `train.py` is used for training our machine with the dataset and the main code - `detect.py` is used to test the training on a new image/video. We also have a sub-code - `datayaml.py` to make the `yaml file` which contains the classes.

Terminal commands are as follows:

Training: 
```
python train.py --img 640 --batch 16 --epochs 50 --data ../building_detection/data.yaml --weights yolov5s.pt --workers 1 --project "buildingTraining" --name "yolov5s_size640_epochs50_batch16_small" --exist-ok
```

Detecting (testing): 
```
python detect.py --weights /home/arismita/ML/yolov5_training/yolov5/buildingTraining/yolov5s_size640_epochs50_batch16_small/weights/best.pt --source ~/Downloads/building_multi_colour.jpeg
```

3. Finger Tracker:

We have made a webpage to display the simulation of the drone firing on the soldiers. Keep the file - `soldiers.html` open in the background and then run the handTracker code to control the mouse pointer using your finger.

Terminal command: 
```
python handTracker.py
```

4. Audio-To-Text/ NLP:

We will have to run the Python code `likithacanyoupleaseshutup.py` and give the audio file from which we want to do keyword extraction as input. The audio file should be in `.wav` format.

Terminal command:
```
python likithacanyoupleaseshutup.py
```

5. Face Detection:

We have 3 main codes under this section. The `cnn_face_encoder.py` file is used to detect faces in each frame of the video and make a pickle file out of it. The `clusterFaces.py` file is used to group similar faces into classes and make another pickle file of clustered faces. The `comperator_actual.py` file is used to compare the faces in the new video with the faces our machine was trained with (detection). We also have to link the pickle files and `mmod_human_face_detector.dat` at the necessary places.

Terminal commands are as follows:

Testing: 
```
python cnn_face_encoder.py -i ./vdos/train.mp4 -m ./mmod_human_face_detector.dat -e ./encodings/face_gautam.pickle -d ./faceDump/face_gautam
```

Clustering: 
```
python clusterFaces.py -e encodings/face_gautam.pickle -d 1 -o encodings/face_gautam-clustered.pickle
```

Detecting (testing): 
```
python3 comperator_actual.py -i vdos/test.mp4 -m mmod_human_face_detector.dat -e encodings/face_gautam-clustered.pickle
```
## Applications
General Applications:

1. Can be applied in military situations like the one described in the video
2. Can also be extended to situations like kidnapping or abduction

Specific Applications:

1. Morse Code: It uses blink detection which can be extended to drowsiness detection which can be installed in vehicles and study rooms.

2. Object Detection: We are detecting buildings in this scenario but the same model can be used to detect various other objects as the situation demands in different scenarios.

3. Finger Tracker: It is being used to remotely control drones in our scenario. It can be extended to sign language recognition, biometric authentication, healthcare, and rehabilitation.

4. Audio To Text/NLP: It can help assist senior citizens who may find it hard to type due to hand tremors. It will be easier for them to record their voice and have it converted to text.

5. Face Detection: It is being used to distinguish between enemies and comrades in this case but the same model can be used for proxy detection in educational institutions and workplaces as face detection for marking attendance would be a strict system which would be hard to cheat.

## Further Improvements
1. Face Detection: Our face clustering code groups faces into different classes. There can be cases where the side profile and the front profile of the same person are put into different classes and we manually have to group them as the same person. We would like to improve on this so that the machine itself can group the front and side profiles of a person and put them into a single class.

2. Finger Tracker: It could have had a better UI and a faster response rate. However, we chose to go with a very simple format because we felt that implementing ML matters more in the hackathon than what we are doing on the front end.

## Demo Video
[Click here](https://www.youtube.com/watch?v=e6uIOivYKd4)
